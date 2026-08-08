"""Native LDAP/AD bind authentication (``INQTRIX_AUTH_MODE=ldap``).

A THIN wrapper over an existing directory via ``ldap3`` (pure-Python, no
OpenLDAP C deps) — Inqtrix does not run an LDAP server, it binds to yours.
The classic search-then-bind: bind as a read-only service account, search
for the user entry (the username is ``escape_filter_chars``-escaped to
defeat LDAP injection), then re-bind as the found DN with the user's
password to verify it. Mapped attributes (email/display/id) and optional
admin-group membership produce an :class:`LdapIdentity`.

The provider subclasses :class:`~inqtrix.auth.oidc.OidcAuthProvider` and
reuses the session/CSRF/PAT/user-mirror machinery verbatim under the
synthetic issuer ``"ldap"``. Constructor-First: every
connection value arrives as an argument; only the settings bridge reads
the environment.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from ldap3 import SUBTREE, SYNC, Connection, Server, Tls
from ldap3.utils.conv import escape_filter_chars

from inqtrix.auth.oidc import OidcAuthProvider

if TYPE_CHECKING:
    from inqtrix.auth.lifecycle import UserLifecycleService

    from inqtrix.auth.directory import UserDirectory
    from inqtrix.auth.invitations import RegistrationGate
    from inqtrix.auth.pat import PatService, PatVerifier
    from inqtrix.auth.principal import AuthMode
    from inqtrix.auth.ratelimit import LoginRateLimiter
    from inqtrix.auth.sessions import FlowStore, SessionStore

log = logging.getLogger("inqtrix")

LDAP_ISSUER = "ldap"
"""Synthetic issuer anchoring LDAP identities in the shared (issuer,
subject) space."""


class LdapError(RuntimeError):
    """Uniform LDAP auth failure (unknown user / wrong password / disabled
    bind all collapse here so the client cannot distinguish them)."""


@dataclass(frozen=True)
class LdapIdentity:
    """A verified LDAP user.

    Attributes:
        subject: Stable id (``id_attr`` value, e.g. ``entryUUID``; falls
            back to the user DN). The identity anchor with
            :data:`LDAP_ISSUER`.
        email: Mapped email (falls back to the login username).
        display_name: Mapped display name (falls back to email).
        is_admin: Whether the user is a member of the configured admin
            group (drives the instance-admin grant on login).
    """

    subject: str
    email: str
    display_name: str
    is_admin: bool


def _first(values: object) -> str | None:
    """First scalar of an ldap3 attribute value (list or scalar) or None.

    Binary attribute values are coerced to a stable, printable string —
    never the ``b'...'`` repr that ``str(bytes)`` would otherwise leak into
    the identity anchor. Active Directory's ``objectGUID`` arrives as raw
    bytes whenever no server schema is loaded (the default here, and always
    under the test mock), so a 16-byte value renders as its canonical GUID
    (mixed-endian, matching what AD tooling shows); any other binary value
    falls back to hex.
    """
    if isinstance(values, (list, tuple)):
        value = values[0] if values else None
    else:
        value = values
    if value is None or value == "":
        return None
    if isinstance(value, bytes):
        return str(uuid.UUID(bytes_le=value)) if len(value) == 16 else value.hex()
    return str(value)


_DN_HEX_ESCAPE = re.compile(r"\\([0-9A-Fa-f]{2})")


def _normalize_group_dn(dn: str) -> str:
    """Canonicalise a group DN for equality comparison.

    Directory servers render the same DN in superficially different ways:
    case (``CN=`` vs ``cn=``), RFC 4514 hex escapes (``\\20`` for a space),
    and optional whitespace around the RDN separators (Active Directory
    emits ``cn=admins, ou=groups, ...``). String-equality on the raw value
    would then deny a legitimate admin their role (a fail-CLOSED defect, not
    an escalation). We fold all three away with one cheap pass rather than a
    full RFC-4514 parser (ldap3's ``parse_dn``/``safe_dn`` reject the
    AD-spacing form outright and do not unescape ``\\HH``): decode the hex
    escapes, collapse whitespace hugging ``,`` and ``=``, then case-fold.
    The same normalisation is applied to both sides, so an escaped separator
    can never widen the match to a different group.
    """
    decoded = _DN_HEX_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), dn)
    return re.sub(r"\s*([,=])\s*", r"\1", decoded).strip().lower()


class LdapClient:
    """Search-then-bind verifier against an external LDAP/AD directory.

    Args:
        url: ``ldap://host:389`` or ``ldaps://host:636``.
        bind_dn: Service-account DN used to search (least-privilege, e.g.
            a read-only account).
        bind_password: Service-account password.
        user_search_base: Base DN for the user search.
        user_search_filter: Filter with a ``{username}`` placeholder, e.g.
            ``(uid={username})``; the username is escaped before formatting.
        email_attr / display_name_attr / id_attr: Attribute names mapped
            to the identity (with fallbacks).
        admin_group_dn: Optional group DN; members get instance-admin.
        start_tls: Issue StartTLS on an ``ldap://`` connection.
        ca_cert: Optional PEM CA bundle path for ldaps/StartTLS.
        validate_cert: Verify the server certificate (default True). Set
            False only for trusted-network dev; a WARNING is logged.
        connection_factory / client_strategy: Test seams. The factory, if
            given, returns a (possibly mock) ``Connection`` for ``(user,
            password)``; ``client_strategy`` switches the real connection
            strategy (tests use ``MOCK_SYNC`` with a seed).
    """

    def __init__(
        self,
        *,
        url: str,
        bind_dn: str,
        bind_password: str,
        user_search_base: str,
        user_search_filter: str = "(uid={username})",
        email_attr: str = "mail",
        display_name_attr: str = "cn",
        id_attr: str = "entryUUID",
        admin_group_dn: str = "",
        start_tls: bool = False,
        ca_cert: str = "",
        validate_cert: bool = True,
        connection_factory: "Callable[[str, str], Connection] | None" = None,
        client_strategy: object = SYNC,
        _seed: "Callable[[Connection], None] | None" = None,
    ) -> None:
        self._url = url
        self._bind_dn = bind_dn
        self._bind_password = bind_password
        self._user_search_base = user_search_base
        self._user_search_filter = user_search_filter
        self._email_attr = email_attr
        self._display_name_attr = display_name_attr
        self._id_attr = id_attr
        self._admin_group_dn = _normalize_group_dn(admin_group_dn)
        self._start_tls = start_tls
        self._ca_cert = ca_cert
        self._validate_cert = validate_cert
        self._connection_factory = connection_factory
        self._client_strategy = client_strategy
        self._seed = _seed
        if not validate_cert:
            log.warning(
                "LDAP-TLS-Zertifikatspruefung ist deaktiviert "
                "(INQTRIX_LDAP_TLS_VALIDATE=false) — nur in vertrauten Netzen."
            )

    def _connect(self, user: str, password: str) -> "Connection":
        if self._connection_factory is not None:
            conn = self._connection_factory(user, password)
        else:
            import ssl as _ssl

            tls = None
            if self._url.lower().startswith("ldaps://") or self._start_tls:
                tls = Tls(
                    validate=(
                        _ssl.CERT_REQUIRED
                        if self._validate_cert
                        else _ssl.CERT_NONE
                    ),
                    ca_certs_file=self._ca_cert or None,
                )
            server = Server(self._url, use_ssl=self._url.lower().startswith("ldaps://"), tls=tls)
            conn = Connection(
                server,
                user=user,
                password=password,
                client_strategy=self._client_strategy,
                auto_bind=False,
            )
        if self._seed is not None:
            self._seed(conn)
        if self._start_tls and not self._url.lower().startswith("ldaps://"):
            conn.start_tls()
        if not conn.bind():
            raise LdapError("Ungueltige Anmeldedaten.")
        return conn

    def authenticate(self, username: str, password: str) -> LdapIdentity:
        """Search-then-bind; return the identity or raise :class:`LdapError`.

        A wrong password is indistinguishable from an unknown user. A
        FAILED SERVICE bind is logged distinctly (operator misconfig, not
        a user error) but still surfaces the same uniform error.
        """
        if not username or not password:
            raise LdapError("Ungueltige Anmeldedaten.")
        # 1) Service bind + search (escaped username defeats LDAP injection).
        try:
            service = self._connect(self._bind_dn, self._bind_password)
        except LdapError:
            log.warning(
                "LDAP-Service-Bind fehlgeschlagen — Konfiguration pruefen "
                "(INQTRIX_LDAP_BIND_DN/_PASSWORD)."
            )
            raise LdapError("Ungueltige Anmeldedaten.")
        safe = escape_filter_chars(username)
        search_filter = self._user_search_filter.format(username=safe)
        attrs = [
            self._email_attr,
            self._display_name_attr,
            self._id_attr,
            "memberOf",
        ]
        service.search(
            search_base=self._user_search_base,
            search_filter=search_filter,
            search_scope=SUBTREE,
            attributes=attrs,
        )
        entries = list(service.entries)
        service.unbind()
        if len(entries) != 1:
            if len(entries) > 1:
                log.warning(
                    "LDAP-Suche lieferte %d Treffer fuer einen Benutzer — "
                    "Filter praezisieren.",
                    len(entries),
                )
            raise LdapError("Ungueltige Anmeldedaten.")
        entry = entries[0]
        user_dn = entry.entry_dn
        values = entry.entry_attributes_as_dict
        # 2) Re-bind as the user to verify the password.
        user_conn = self._connect(user_dn, password)
        user_conn.unbind()
        # 3) Map attributes with fallbacks; identity anchor = id_attr or DN.
        email = _first(values.get(self._email_attr)) or username
        display_name = _first(values.get(self._display_name_attr)) or email
        subject = _first(values.get(self._id_attr)) or user_dn
        member_of = values.get("memberOf") or []
        is_admin = bool(self._admin_group_dn) and self._admin_group_dn in {
            _normalize_group_dn(str(group)) for group in member_of
        }
        return LdapIdentity(
            subject=subject,
            email=email,
            display_name=display_name,
            is_admin=is_admin,
        )


class LdapAuthProvider(OidcAuthProvider):
    """LDAP bind provider on the shared session-cookie machinery.

    Args:
        ldap_client: The search-then-bind verifier.
        first_login_owner: When True (default) the first authenticated
            user becomes the instance admin if none exists yet.

    The remaining arguments are the session collaborators the base takes;
    ``client`` is omitted (no IdP) and ``flows`` is supplied only to
    satisfy the base constructor.
    """

    def __init__(
        self,
        *,
        ldap_client: LdapClient,
        first_login_owner: bool = True,
        sessions: "SessionStore",
        flows: "FlowStore",
        users: "UserDirectory | None" = None,
        session_secret: str,
        session_max_age_seconds: int,
        secure_cookies: bool = True,
        pats: "PatVerifier | None" = None,
        pat_service: "PatService | None" = None,
        registration_gate: "RegistrationGate | None" = None,
        login_rate_limiter: "LoginRateLimiter | None" = None,
        trusted_proxy_hops: int = 1,
        lifecycle: "UserLifecycleService | None" = None,
    ) -> None:
        super().__init__(
            client=None,
            sessions=sessions,
            flows=flows,
            users=users,
            session_secret=session_secret,
            session_max_age_seconds=session_max_age_seconds,
            secure_cookies=secure_cookies,
            pats=pats,
            pat_service=pat_service,
            registration_gate=registration_gate,
            lifecycle=lifecycle,
        )
        self.ldap_client = ldap_client
        self.first_login_owner = first_login_owner
        self.login_rate_limiter = login_rate_limiter
        self.trusted_proxy_hops = trusted_proxy_hops

    @property
    def mode(self) -> "AuthMode":
        """``"ldap"``."""
        return "ldap"

"""Domain whitelists for source tiering."""

from __future__ import annotations


LANG_NAMES: dict[str, str] = {
    "de": "Deutsch", "en": "Englisch", "fr": "Franzoesisch",
    "es": "Spanisch", "it": "Italienisch", "pt": "Portugiesisch",
}

LOW_QUALITY_DOMAINS: list[str] = [
    "-pinterest.com",
    "-pinterest.de",
    "-quora.com",
    "-tiktok.com",
    "-reddit.com",
    "-medium.com",
    "-gutefrage.net",
    "-aitoolsone.com",
    "-aipressroom.com",
    "-aitoolsrecap.com",
    "-aitoolly.com",
    "-tldl.io",
    "-toolscompare.ai",
    "-opentools.ai",
    "-kersai.com",
    "-chatfin.ai",
    "-coaio.com",
]

PRIMARY_REGULATOR_DOMAINS: set[str] = {
    # Bundesregierung & Ministerien
    "bundesregierung.de",
    "bundesgesundheitsministerium.de",
    "bmg.bund.de",
    "bundestag.de",
    "dip.bundestag.de",            # Dokumentations- und Informationssystem für Parlamentsmaterialien
    "das-parlament.de",            # Bundestag-eigenes Wochenmagazin
    "gesetze-im-internet.de",
    "gkv-spitzenverband.de",
    "bundesrechnungshof.de",       # unabhängige Finanzkontrolle des Bundes
    "abgeordnetenwatch.de",        # parlamentarische Transparenz / amtliche Daten
    "bafin.de",                    # Bundesanstalt für Finanzdienstleistungsaufsicht
    "bundesbank.de",               # Deutsche Bundesbank (Primary central bank)
    # EU & internationale Institutionen
    "ec.europa.eu",
    "europa.eu",
    "commission.europa.eu",
    "eurostat.ec.europa.eu",
    "data.europa.eu",
    "ecb.europa.eu",               # European Central Bank
    "esma.europa.eu",              # European Securities and Markets Authority
    "eba.europa.eu",               # European Banking Authority
    "who.int",
    "oecd.org",
    "un.org",
    "unctad.org",                  # UN Conference on Trade & Development
    "fao.org",                     # UN Food and Agriculture Organization
    "iaea.org",                    # International Atomic Energy Agency
    "wto.org",                     # World Trade Organization
    "bis.org",                     # Bank for International Settlements
    "iea.org",                     # International Energy Agency
    "irena.org",                   # International Renewable Energy Agency
    "weforum.org",                 # World Economic Forum (data/reports)
    # US Bundesbehörden
    "sec.gov",
    "federalreserve.gov",
    "bls.gov",
    "bea.gov",
    "census.gov",
    "energy.gov",
    "eia.gov",
    "ftc.gov",
    "justice.gov",
    "nist.gov",
    "data.gov",
    "cdc.gov",                     # Centers for Disease Control
    "nih.gov",                     # National Institutes of Health (covers ncbi.nlm.nih.gov, pubmed)
    "fda.gov",                     # Food and Drug Administration
    "hhs.gov",                     # Health & Human Services
    "treasury.gov",                # US Treasury
    "congress.gov",                # US Congress
    "gao.gov",                     # Government Accountability Office
    "cbo.gov",                     # Congressional Budget Office
    "whitehouse.gov",
    "cftc.gov",                    # Commodity Futures Trading Commission
    "dol.gov",                     # Department of Labor
    "irs.gov",                     # Internal Revenue Service
    "epa.gov",                     # Environmental Protection Agency
    "noaa.gov",                    # National Oceanic and Atmospheric Administration
    "weather.gov",
    "usgs.gov",                    # US Geological Survey
    "nasa.gov",
    "faa.gov",                     # Federal Aviation Administration
    "dot.gov",                     # Department of Transportation
    "ed.gov",                      # Department of Education
    "gpo.gov",                     # Government Publishing Office
    # US Federal Reserve Banks (12 districts) — research often appears under fred.stlouisfed.org etc.
    "newyorkfed.org",
    "bostonfed.org",
    "philadelphiafed.org",
    "clevelandfed.org",
    "richmondfed.org",
    "frbatlanta.org",
    "chicagofed.org",
    "stlouisfed.org",              # covers fred.stlouisfed.org, research.stlouisfed.org
    "minneapolisfed.org",
    "kansascityfed.org",
    "dallasfed.org",
    "frbsf.org",                   # Federal Reserve Bank of San Francisco
    # Other major central banks
    "bankofengland.co.uk",
    "snb.ch",                      # Swiss National Bank
    "boj.or.jp",                   # Bank of Japan
    "rba.gov.au",                  # Reserve Bank of Australia
    "bankofcanada.ca",             # Bank of Canada
    "rbi.org.in",                  # Reserve Bank of India
    # UK government & regulators
    "ons.gov.uk",                  # Office for National Statistics
    "gov.uk",                      # umbrella for UK government
    "fca.org.uk",                  # Financial Conduct Authority
    "nao.org.uk",                  # National Audit Office
    # Canadian government
    "canada.ca",                   # Government of Canada portal
    "statcan.gc.ca",               # Statistics Canada
    # German statistics
    "destatis.de",
    # International financial institutions
    "worldbank.org",
    "imf.org",
}

PRIMARY_OFFICIAL_COMPANY_DOMAINS: set[str] = {
    # AI labs (official channels)
    "openai.com",
    "anthropic.com",
    "ai.google",
    "blog.google",
    "deepmind.google",
    "ai.meta.com",                 # Meta AI Research (offizielles Lab)
    "research.facebook.com",       # Meta Research legacy domain
    # Microsoft
    "news.microsoft.com",
    "blogs.microsoft.com",
    "techcommunity.microsoft.com",
    # IBM
    "newsroom.ibm.com",
    "research.ibm.com",
    # Meta corporate
    "investor.atmeta.com",
    "about.fb.com",
    # Tesla / Nvidia / Apple investor + press
    "ir.tesla.com",
    "assets-ir.tesla.com",
    "tesla-cdn.thron.com",
    "investor.nvidia.com",
    "nvidianews.nvidia.com",
    "investor.apple.com",
    "abc.xyz",                     # Alphabet
    "sundar.google",
    # Amazon
    "aboutamazon.com",             # Amazon press / newsroom
    "ir.aboutamazon.com",
    # Intel
    "newsroom.intel.com",
    "blogs.intel.com",
    # Oracle (offizielle Tech-Blogs, Produktankündigungen)
    "blogs.oracle.com",
    "oracle.com",                  # corporate / IR
}

PRIMARY_ACADEMIC_INSTITUTION_DOMAINS: set[str] = {
    # US Top-Universities
    "hai.stanford.edu",
    "stanford.edu",
    "mit.edu",
    "technologyreview.mit.edu",
    "berkeley.edu",
    "harvard.edu",
    "yale.edu",
    "princeton.edu",
    "columbia.edu",
    "uchicago.edu",
    "upenn.edu",
    "cornell.edu",
    "jhu.edu",                     # Johns Hopkins
    "caltech.edu",
    "nyu.edu",
    "duke.edu",
    "ucla.edu",
    "umich.edu",
    "ucdavis.edu",                 # covers libguides.law.ucdavis.edu
    "gatech.edu",                  # Georgia Tech
    "cmu.edu",                     # Carnegie Mellon
    "ucsd.edu",
    "brown.edu",
    "dartmouth.edu",
    "northwestern.edu",
    "tufts.edu",                   # covers digitalplanet.tufts.edu, now.tufts.edu
    # UK / Europe
    "ox.ac.uk",                    # Oxford
    "cam.ac.uk",                   # Cambridge
    "imperial.ac.uk",
    "lse.ac.uk",
    "ucl.ac.uk",
    "ed.ac.uk",                    # Edinburgh
    "kcl.ac.uk",                   # King's College London
    # Continental Europe research institutions
    "ethz.ch",                     # ETH Zürich
    "epfl.ch",                     # EPFL Lausanne
    "tum.de",                      # TU München
    "lmu.de",                      # LMU München
    "uni-heidelberg.de",
    "fu-berlin.de",
    "hu-berlin.de",
    "rwth-aachen.de",
    "kit.edu",                     # KIT Karlsruhe
    "mpg.de",                      # Max-Planck-Gesellschaft
    "fraunhofer.de",
    "helmholtz.de",
    "leibniz-gemeinschaft.de",
    "sciencespo.fr",
    "ens.fr",
    "inria.fr",                    # INRIA (French CS research)
    "cnrs.fr",                     # CNRS (French scientific research)
}

PRIMARY_ACADEMIC_PUBLISHER_DOMAINS: set[str] = {
    # Top medical / general science journals
    "nejm.org",                    # New England Journal of Medicine
    "thelancet.com",
    "jamanetwork.com",             # JAMA Network
    "bmj.com",                     # British Medical Journal
    "cell.com",                    # Cell Press
    "pnas.org",                    # Proceedings of the National Academy of Sciences
    "plos.org",                    # Public Library of Science
    # Top publishers: deliberately the journal-database subdomain, NOT the
    # publisher's apex domain. Reason: apex-suffix matches (e.g. ``ieee.org``)
    # would also tier the journalistic trade magazines (``spectrum.ieee.org``)
    # as Primary, which is wrong — those are Mainstream journalism.
    "ieeexplore.ieee.org",         # IEEE peer-reviewed papers (not spectrum.ieee.org)
    "dl.acm.org",                  # ACM Digital Library (not cacm.acm.org)
    "journals.sagepub.com",
    "onlinelibrary.wiley.com",
    "link.springer.com",
    "sciencedirect.com",           # Elsevier journals
    "jstor.org",
    "tandfonline.com",             # Taylor & Francis
    "annualreviews.org",
    "academic.oup.com",            # Oxford University Press journals
    "cambridge.org",               # Cambridge University Press journals
    "ams.org",                     # American Mathematical Society
    "journals.aps.org",            # American Physical Society
    "pubs.acs.org",                # American Chemical Society
    "pubs.aip.org",                # American Institute of Physics
    "pubs.rsc.org",                # Royal Society of Chemistry
    "aaai.org",                    # Association for the Advancement of AI
    "openreview.net",              # peer review for top ML conferences
    # Preprint servers (Primary academic by community convention)
    "biorxiv.org",
    "medrxiv.org",
    "ssrn.com",                    # Social Science Research Network
    # NIH / NCBI search infrastructure (suffix-covered by nih.gov)
}

PRIMARY_SOURCE_DOMAINS: set[str] = (
    PRIMARY_REGULATOR_DOMAINS
    | PRIMARY_OFFICIAL_COMPANY_DOMAINS
    | PRIMARY_ACADEMIC_INSTITUTION_DOMAINS
    | PRIMARY_ACADEMIC_PUBLISHER_DOMAINS
    | {
        "nature.com",
        "science.org",
        "arxiv.org",
    }
)

MAINSTREAM_SOURCE_DOMAINS: set[str] = {
    # -------------------------------------------------------------------
    # Deutsche Medien (Tagespresse + Wochenmagazine)
    # -------------------------------------------------------------------
    "aerzteblatt.de",
    "aerztezeitung.de",            # Mainstream-Fachmedium Medizin
    "deutsche-apotheker-zeitung.de",  # Mainstream-Fachmedium Pharma
    "deutschlandfunk.de",
    "tagesspiegel.de",
    "handelsblatt.com",
    "spiegel.de",
    "zdfheute.de",
    "heute.de",                    # ZDF heute (Hauptdomain)
    "tagesschau.de",
    "ard.de",
    "stern.de",
    "zeit.de",
    "faz.net",
    "sueddeutsche.de",
    "welt.de",
    "n-tv.de",
    "focus.de",                    # Mainstream-News-Magazin
    "taz.de",                      # Mainstream-Tageszeitung
    "ntv.de",
    "rnd.de",                      # RedaktionsNetzwerk Deutschland
    "fr.de",                       # Frankfurter Rundschau
    "merkur.de",
    # DE Tech-Medien (Fachjournalismus, hohe Reputation)
    "heise.de",                    # IT-News-Marktführer DE
    "golem.de",
    "t3n.de",
    "chip.de",
    "computerbild.de",
    "winfuture.de",
    "computerwoche.de",
    "it-business.de",
    # DE Wirtschaft / Finance
    "manager-magazin.de",
    "wiwo.de",                     # WirtschaftsWoche
    "capital.de",
    "boerse-online.de",
    "finanzen.net",
    "dasinvestment.com",
    "finanztip.de",                # unabhängige Verbraucher-Finanzberatung
    # Datenservices
    "de.statista.com",
    "statista.com",
    # Schweiz (deutschsprachig, hohe Reputation)
    "nzz.ch",                      # Neue Zürcher Zeitung
    "srf.ch",                      # Schweizer Radio und Fernsehen
    "blick.ch",
    "tagesanzeiger.ch",
    "cash.ch",
    # Österreich
    "derstandard.at",
    "diepresse.com",
    "orf.at",
    "kurier.at",
    "wienerzeitung.at",
    # -------------------------------------------------------------------
    # Internationale Nachrichtenagenturen
    # -------------------------------------------------------------------
    "reuters.com",
    "apnews.com",
    "ap.org",
    "afp.com",                     # Agence France-Presse
    "afpforum.com",
    "dpa.com",                     # Deutsche Presse-Agentur
    "kyodonews.net",               # Kyodo News (Japan)
    "yna.co.kr",                   # Yonhap News (Korea)
    "ansa.it",                     # ANSA (Italy)
    "efe.com",                     # EFE (Spain)
    # -------------------------------------------------------------------
    # US/UK Mainstream
    # -------------------------------------------------------------------
    "nytimes.com",
    "washingtonpost.com",
    "bbc.com",
    "bbc.co.uk",
    "theguardian.com",
    "cnn.com",
    "abcnews.go.com",
    "abcnews.com",
    "nbcnews.com",
    "cbsnews.com",
    "npr.org",
    "pbs.org",
    "latimes.com",                 # Los Angeles Times
    "chicagotribune.com",
    "bostonglobe.com",
    "usatoday.com",
    "politico.com",
    "axios.com",
    "theatlantic.com",
    "vox.com",
    "newyorker.com",
    "propublica.org",              # investigative non-profit
    "democracynow.org",            # independent broadcast journalism
    "csmonitor.com",               # Christian Science Monitor
    "jacobin.com",                 # established US political magazine
    # UK
    "telegraph.co.uk",
    "thetimes.com",
    "thetimes.co.uk",              # The Times of London
    "independent.co.uk",
    "the-independent.com",
    "news.sky.com",
    "channel4.com",
    "itv.com",
    "newscientist.com",
    "newsnow.co.uk",
    # Ireland
    "irishtimes.com",
    "rte.ie",                      # RTE (Irish public broadcaster)
    "siliconrepublic.com",         # Irish tech news
    # -------------------------------------------------------------------
    # Westeuropa
    # -------------------------------------------------------------------
    "politico.eu",                 # Politico Europe
    "euronews.com",
    "dw.com",                      # Deutsche Welle (intl. DE-Public-Service)
    # France
    "lemonde.fr",
    "lefigaro.fr",
    "liberation.fr",
    "lesechos.fr",
    "lepoint.fr",
    "francetvinfo.fr",
    # Italy
    "corriere.it",
    "repubblica.it",
    "ilsole24ore.com",
    "lastampa.it",
    # Spain
    "elpais.com",
    "elmundo.es",
    "abc.es",
    "lavanguardia.com",
    # Netherlands
    "nrc.nl",                      # NRC Handelsblad
    "volkskrant.nl",
    "nos.nl",                      # NOS (Dutch public broadcaster)
    # Belgium
    "standaard.be",
    "lesoir.be",
    # Scandinavia
    "dn.se",                       # Dagens Nyheter (Sweden)
    "svt.se",                      # SVT (Swedish public broadcaster)
    "aftonbladet.se",
    "nrk.no",                      # NRK (Norwegian public broadcaster)
    "dr.dk",                       # DR (Danish public broadcaster)
    "politiken.dk",
    "yle.fi",                      # Yle (Finnish public broadcaster)
    "hs.fi",                       # Helsingin Sanomat
    # -------------------------------------------------------------------
    # Wirtschaftsmedien international
    # -------------------------------------------------------------------
    "bloomberg.com",
    "businessweek.com",            # Bloomberg Businessweek
    "ft.com",
    "wsj.com",
    "barrons.com",                 # Barron's (Dow Jones)
    "marketwatch.com",
    "economist.com",
    "hbr.org",                     # Harvard Business Review
    "businessinsider.com",
    "fortune.com",
    "forbes.com",
    "cnbc.com",
    "investors.com",               # Investor's Business Daily
    "ibd.com",
    "economictimes.com",
    "morningstar.com",             # Investment-Research / Mainstream-Marktanalyse
    "zacks.com",                   # Investment-Research / Mainstream-Marktanalyse
    "nasdaq.com",                  # offizielle Nasdaq News
    "finance.yahoo.com",           # Yahoo Finance editorial
    "benzinga.com",
    "investing.com",
    "tradingview.com",
    "barchart.com",
    "nerdwallet.com",              # Verbraucher-Finanzen
    # -------------------------------------------------------------------
    # Tech-Medien (international)
    # -------------------------------------------------------------------
    "techcrunch.com",
    "arstechnica.com",
    "theverge.com",
    "wired.com",
    "technologyreview.com",
    "spectrum.ieee.org",
    "infoq.com",
    "engadget.com",
    "gizmodo.com",
    "mashable.com",
    "venturebeat.com",
    "cnet.com",
    "techradar.com",
    "hothardware.com",
    "tomshardware.com",
    "anandtech.com",
    "theregister.com",             # The Register (UK enterprise IT)
    "crn.com",                     # IT-Trade-Publikation
    "informationweek.com",
    "eweek.com",
    "sdtimes.com",                 # Software Development Times
    "diginomica.com",              # Enterprise-IT-Journalismus
    "blocksandfiles.com",          # Storage-Industrie-Journalismus
    "thenewstack.io",
    # Cybersecurity (etablierter Fachjournalismus)
    "krebsonsecurity.com",
    "bleepingcomputer.com",
    "darkreading.com",
    "helpnetsecurity.com",
    "securityweek.com",
    "cybernews.com",
    "bankinfosecurity.com",
    # -------------------------------------------------------------------
    # Asien Mainstream
    # -------------------------------------------------------------------
    # Japan
    "nikkei.com",                  # covers asia.nikkei.com
    "asahi.com",
    "yomiuri.co.jp",
    "mainichi.jp",
    "japantimes.co.jp",
    "nhk.or.jp",
    "nippon.com",
    # China / Hong Kong / Taiwan
    "scmp.com",                    # South China Morning Post
    "36kr.com",                    # covers eu.36kr.com
    "caixinglobal.com",            # Caixin Global (Chinese business)
    # Singapore / SEA
    "straitstimes.com",
    "channelnewsasia.com",
    "todayonline.com",
    "bangkokpost.com",
    "thestar.com.my",
    "nst.com.my",
    # Philippines
    "abs-cbn.com",
    "gmanetwork.com",
    "philstar.com",
    # Korea
    "koreaherald.com",
    "koreatimes.co.kr",
    "joongangdaily.joins.com",
    "sedaily.com",                 # covers en.sedaily.com (Seoul Economic Daily)
    # India
    "thehindu.com",
    "hindustantimes.com",
    "indianexpress.com",
    "timesofindia.indiatimes.com",
    "indiatimes.com",
    "ndtv.com",
    "livemint.com",
    "business-standard.com",
    "thehindubusinessline.com",
    # -------------------------------------------------------------------
    # Naher Osten / MENA
    # -------------------------------------------------------------------
    "aljazeera.com",
    "aljazeera.net",
    "haaretz.com",
    "timesofisrael.com",
    "jpost.com",                   # Jerusalem Post
    "arabnews.com",
    "thenationalnews.com",         # The National (UAE)
    "gulfnews.com",
    # -------------------------------------------------------------------
    # Afrika
    # -------------------------------------------------------------------
    "mg.co.za",                    # Mail & Guardian (South Africa)
    "dailymaverick.co.za",
    "news24.com",
    "businesslive.co.za",
    "nation.co.ke",                # Daily Nation (Kenya)
    "premiumtimesng.com",          # Premium Times (Nigeria)
    # -------------------------------------------------------------------
    # Lateinamerika
    # -------------------------------------------------------------------
    "folha.uol.com.br",
    "oglobo.globo.com",
    "valor.globo.com",
    "clarin.com",                  # Argentina
    "lanacion.com.ar",
    "infobae.com",
    "reforma.com",                 # Mexico
    "eluniversal.com.mx",
    "eleconomista.com.mx",
    # -------------------------------------------------------------------
    # Australien / Neuseeland
    # -------------------------------------------------------------------
    "abc.net.au",                  # ABC Australia (different from US ABC)
    "smh.com.au",                  # Sydney Morning Herald
    "theage.com.au",
    "afr.com",                     # Australian Financial Review
    "theaustralian.com.au",
    "stuff.co.nz",
    "rnz.co.nz",                   # Radio NZ
    "nzherald.co.nz",
    # -------------------------------------------------------------------
    # Wissenschaft (allgemein, journalistisch aufbereitet)
    # -------------------------------------------------------------------
    "nature.com",
    "science.org",
    "scientificamerican.com",
    "newscientist.com",
    "phys.org",
    "sciencenews.org",
    "livescience.com",
    "arxiv.org",
    # -------------------------------------------------------------------
    # Polling / Reference / Datenjournalismus
    # -------------------------------------------------------------------
    "pewresearch.org",             # Pew Research (highly cited, methodologically transparent)
    "yougov.com",                  # YouGov polling
    "en.wikipedia.org",            # English Wikipedia (Quervergleich; nicht Primärquelle)
    "wikipedia.org",
    "britannica.com",              # Encyclopædia Britannica
    "snopes.com",                  # Fact-checking
    "factcheck.org",
    "politifact.com",
    "fullfact.org",                # UK fact-checking
    "correctiv.org",               # DE investigative / fact-checking
}

STAKEHOLDER_SOURCE_DOMAINS: set[str] = {
    # -------------------------------------------------------------------
    # Selbstverwaltung & Verbände im Gesundheitswesen (DE)
    # -------------------------------------------------------------------
    "kzbv.de",
    "bzaek.de",
    "kbv.de",                      # Kassenärztliche Bundesvereinigung
    "vdek.com",
    "aok.de",
    "pkv.de",                      # PKV-Verband
    "deutscher-pflegerat.de",
    "physio-deutschland.de",
    "marburger-bund.de",
    "dkgev.de",                    # Deutsche Krankenhausgesellschaft
    "arbeitgeber.de",              # BDA
    # -------------------------------------------------------------------
    # Parteien (Positionen offizieller Parteiorganisationen)
    # -------------------------------------------------------------------
    "cdu.de",
    "spd.de",
    "gruene.de",
    "fdp.de",
    "linke.de",
    "csu.de",
    # -------------------------------------------------------------------
    # Think Tanks (US) — analytisch hochwertig, aber Advocacy-Standpunkt
    # -------------------------------------------------------------------
    "brookings.edu",
    "rand.org",
    "cfr.org",                     # Council on Foreign Relations
    "carnegieendowment.org",
    "csis.org",                    # Center for Strategic & International Studies
    "atlanticcouncil.org",
    "hudson.org",
    "aei.org",                     # American Enterprise Institute
    "heritage.org",
    "cato.org",
    "urban.org",                   # Urban Institute
    "piie.com",                    # Peterson Institute for International Economics
    "americanactionforum.org",
    "bipartisanpolicy.org",
    "openmarketsinstitute.org",
    "datainnovation.org",
    "ifstudies.org",               # Institute for Family Studies
    "hinrichfoundation.com",       # Trade policy
    "brennancenter.org",           # Brennan Center for Justice
    "newamerica.org",
    "epi.org",                     # Economic Policy Institute
    "americanprogress.org",        # Center for American Progress
    "manhattan-institute.org",
    "mercatus.org",                # Mercatus Center
    "ssrc.org",                    # Social Science Research Council
    # AI / Tech Policy Think Tanks
    "adalovelaceinstitute.org",
    "futureoflife.org",
    "openai.com",                  # already primary, but research papers count both
    "epochai.org",
    "anthropic.com",               # already primary
    # -------------------------------------------------------------------
    # Think Tanks (Europa / International)
    # -------------------------------------------------------------------
    "chathamhouse.org",
    "bruegel.org",
    "ceps.eu",                     # Centre for European Policy Studies
    "ecfr.eu",                     # European Council on Foreign Relations
    "swp-berlin.de",               # Stiftung Wissenschaft und Politik
    "giga-hamburg.de",
    "ifo.de",                      # Ifo Institut
    "diw.de",                      # DIW Berlin
    "iwkoeln.de",                  # Institut der deutschen Wirtschaft
    "bertelsmann-stiftung.de",
    "hertie-school.org",
    "iiss.org",                    # International Institute for Strategic Studies
    "odi.org",                     # Overseas Development Institute (UK)
    # -------------------------------------------------------------------
    # Foundations / Granting Organizations
    # -------------------------------------------------------------------
    "gatesfoundation.org",
    "fordfoundation.org",
    "rockefellerfoundation.org",
    "opensocietyfoundations.org",
    "carnegie.org",
    "macfound.org",                # MacArthur Foundation
    # -------------------------------------------------------------------
    # Consulting firms (Industry-Sicht; Daten gut, Standpunkt vorhanden)
    # -------------------------------------------------------------------
    "mckinsey.com",
    "deloitte.com",
    "kpmg.com",
    "pwc.com",
    "ey.com",
    "bcg.com",                     # Boston Consulting Group
    "bain.com",
    "accenture.com",
    "kearney.com",                 # AT Kearney
    "oliverwyman.com",
    "rolandberger.com",
    "strategyand.pwc.com",
    # -------------------------------------------------------------------
    # Industry / Markt-Research
    # -------------------------------------------------------------------
    "gartner.com",
    "idc.com",
    "forrester.com",
    "oxfordeconomics.com",
    "caixabankresearch.com",       # Caixa Bank Research
    "factset.com",                 # Financial-Data-Anbieter / Marktanalyse
    "spglobal.com",                # S&P Global Ratings & Analytics
    "moodys.com",
    "fitchratings.com",
    # -------------------------------------------------------------------
    # Banks / Asset Managers (Eigenstandpunkt zu Märkten)
    # -------------------------------------------------------------------
    "jpmorgan.com",
    "jpmorganchase.com",
    "goldmansachs.com",
    "morganstanley.com",
    "bankofamerica.com",
    "citigroup.com",
    "ubs.com",
    "credit-suisse.com",           # legacy; kann historische Reports betreffen
    "deutsche-bank.de",
    "blackrock.com",
    "vanguard.com",
    "fidelity.com",
    "alliancebernstein.com",
    "pnc.com",
    "northerntrust.com",
    "schwabnetwork.com",
    "schwab.com",
    # -------------------------------------------------------------------
    # Internationale NGOs / Advocacy (Wikipedia "reliable with attribution")
    # -------------------------------------------------------------------
    "amnesty.org",
    "hrw.org",                     # Human Rights Watch
    "transparency.org",
    "icij.org",                    # International Consortium of Investigative Journalists
}

SOURCE_TIER_WEIGHTS: dict[str, float] = {
    "primary": 1.0,
    "mainstream": 0.8,
    "stakeholder": 0.45,
    "unknown": 0.35,
    "low": 0.1,
}

GENERIC_QUERY_TERMS_DE: set[str] = {
    "soll", "sollen", "sollte", "sollten", "werden", "wird", "wurde",
    "zukuenftig", "kuenftig", "zukunft", "aktuell", "heute",
    "richtung", "diskussion", "diskussionen", "debatte", "debatten",
    "geht", "gehen", "genau", "eigentlich", "was", "waren", "wichtige",
    "wichtigsten", "wichtig", "entwicklung", "entwicklungen", "letzte",
    "letzten", "vergangene", "vergangenen", "tage", "woche", "wochen",
    "news", "nachrichten", "meldungen", "important", "developments",
    "recent", "latest", "last", "days", "week", "weeks",
}

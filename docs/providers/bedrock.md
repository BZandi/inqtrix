# Amazon Bedrock

> File: `src/inqtrix/providers/bedrock.py`

## Scope

`BedrockLLM` is a direct adapter for the Amazon Bedrock Converse API via `boto3`. It is the right choice when your organisation requires data residency on AWS, when Anthropic-via-Bedrock is contractually preferred, or when cross-region inference profiles give you better latency than the Anthropic SaaS.

## When to use it

- You already have Bedrock access and prefer IAM / AWS SSO over a dedicated Anthropic key.
- You need EU cross-region inference (for example `eu.anthropic.claude-sonnet-4-6`).
- You want `AgentRateLimited` on `ThrottlingException` instead of silent retries.

## Constructor

```python
from inqtrix import BedrockLLM


llm = BedrockLLM(
    default_model="eu.anthropic.claude-sonnet-4-6",
    summarize_model="eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    profile_name="my-profile",   # optional AWS profile name
    region_name="eu-central-1",  # optional, defaults to AWS default
)
```

| Parameter | Purpose |
|-----------|---------|
| `default_model` | Bedrock model id (including EU cross-region prefix where applicable). Used for classify fallback, plan, answer, evaluate fallback. |
| `classify_model`, `summarize_model`, `evaluate_model` | Optional per-role overrides. |
| `profile_name` | AWS profile name; passed to `boto3.Session(profile_name=...)`. Use IAM role or SSO credentials. |
| `region_name` | AWS region for Bedrock. |
| `boto3_session` | Inject a pre-built `boto3.Session` instance if you use custom credential providers. |
| `request_max_tokens` | Hard cap on `maxTokens`; enables honest token-utilisation logging. |

## Authentication

Authentication is handled entirely by `boto3`'s credential chain:

1. Explicit `profile_name` or `boto3_session` argument.
2. Environment variables (`AWS_PROFILE`, `AWS_ACCESS_KEY_ID`, ...).
3. IAM role attached to the EC2/ECS/EKS instance.
4. SSO cached credentials.

The provider itself reads no AWS environment variables; it always defers to `boto3`.

## Retry and rate-limit behaviour

- `ThrottlingException`: up to 5 attempts (`_MAX_BEDROCK_ATTEMPTS`) with exponential backoff + jitter. After the last attempt, the error is translated to `AgentRateLimited` so upstream handlers treat it the same as an Anthropic 429.
- Other Bedrock `ClientError` responses: raised as `BedrockAPIError` with the mapped HTTP-ish status code.
- Retries respect the run deadline so retries cannot push past `MAX_TOTAL_SECONDS`.

## Known caveats

- The Bedrock Converse API sets several response fields that the `botocore` model validator enforces strictly; replay cassettes need the full Converse response shape (`usage`, `metrics`, `stopReason`), not just the fields Inqtrix reads (see [Testing strategy](../development/testing-strategy.md)).
- Moto does not currently cover `bedrock-runtime`, which is why the test suite uses `botocore.stub.Stubber` for Bedrock replays.

## Example stacks

Library:

```python
from inqtrix import AgentConfig, BedrockLLM, PerplexitySearch, ResearchAgent

agent = ResearchAgent(AgentConfig(
    llm=BedrockLLM(
        default_model="eu.anthropic.claude-sonnet-4-6",
        region_name="eu-central-1",
    ),
    search=PerplexitySearch(api_key="pplx-..."),
))
```

Server: `examples/webserver_stacks/bedrock_perplexity.py`.

## Related docs

- [Providers overview](overview.md)
- [Anthropic](anthropic.md)
- [Timeouts and errors](../observability/timeouts-and-errors.md)
- [Testing strategy](../development/testing-strategy.md)

# Cloud GPU costs

**Draft, November 2026.** Prices come from third-party listings and change often. Confirm them on
the provider's page before budgeting. Training hours are placeholders until the model owner
gives a real estimate.

## Hourly price and estimated total

Totals cover GPU time only, in US dollars. Storage, disks and data transfer are extra.

| Option | GPU | VRAM | $/hour | 20 h | 100 h | 400 h |
|---|---|---|---|---|---|---|
| AWS `g5.xlarge` Spot | A10G | 24 GB | ~0.25 | 5 | 25 | 100 |
| AWS `g5.xlarge` On-Demand | A10G | 24 GB | ~1.01 | 20 | 101 | 404 |
| Google Cloud `g2-standard-4` Spot | L4 | 24 GB | ~0.30 | 6 | 30 | 120 |
| Google Cloud `g2-standard-4` On-Demand | L4 | 24 GB | ~0.71 | 14 | 71 | 284 |
| RunPod Community Cloud | RTX 4090 | 24 GB | ~0.34 | 7 | 34 | 136 |
| RunPod Secure Cloud | RTX 4090 | 24 GB | ~0.74 | 15 | 74 | 296 |
| Vast.ai marketplace | RTX 4090 | 24 GB | from ~0.14 | 3 | 14 | 56 |

The Google Cloud Spot price is the middle of a $0.28 to $0.33 range. Spot and marketplace prices
move, so treat those totals as a lower bound.

## Credits and limits

| Provider | Credit | Valid for | GPU access |
|---|---|---|---|
| AWS | $100, plus up to $100 more | Up to 12 months | Likely needs the paid plan, plus a quota request |
| Google Cloud | $300 | About 90 days | Needs a paid account, plus a quota request |
| Azure for Students | $100 | 12 months | Not practical: low vCPU quota and blocked GPU sizes |
| RunPod, Vast.ai | None found | Not applicable | Prepaid balance, no quota request |

A paid plan means charges go to the card once the credit runs out.

## Recommendation

AWS Spot `g5.xlarge`, with Google Cloud as the fallback. This is awaiting a team decision.

- The credit lasts long enough for the project.
- Checkpoints already make Spot interruptions safe.
- The code is not tied to AWS, so switching stays cheap.

RunPod and Vast.ai are the cheapest way to start with no quota wait, but their community and
marketplace machines can run on third-party hardware. Only public, license-cleared data may go
there.

## Sources

- [AWS Free Tier announcement](https://aws.amazon.com/about-aws/whats-new/2025/07/aws-free-tier-credits-month-free-plan/)
- [AWS News Blog](https://aws.amazon.com/blogs/aws/aws-free-tier-update-new-customers-can-get-started-and-explore-aws-with-up-to-200-in-credits)
- [Amazon EC2 instance type quotas](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html)
- [A10G listing](https://www.devzero.io/instances/gpu/a10g)
- [g2-standard-4 listing](https://gcloud-compute.com/g2-standard-4.html)
- [Google Cloud GPU listing](https://gpu-cost.pages.dev/provider/gcp/)
- [Google Cloud free trial summary](https://bestusavps.pages.dev/deals/google-cloud-free)
- [Cloud GPU prices, Hostinger](https://www.hostinger.com/tutorials/?p=158956)
- [Azure for Students](https://learn.microsoft.com/ko-kr/azure/education-hub/about-azure-for-students)
- [Azure student subscription limits](https://techcommunity.microsoft.com/discussions/microsoft-learn-for-educators/sku-quota-and-policy-restrictions-on-azure-for-students-and-free-subscriptions/4525160)

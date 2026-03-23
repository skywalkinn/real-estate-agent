---
name: lead_bot_analysis_flags
description: Specific flags and criteria the user wants the AI lead review bot to check
type: project
---

User wants the lead bot to check and flag these specific things on every lead:

1. **Active MLS listing** — is the property currently listed for sale?
2. **Value threshold: $500,000** — flag if estimated value is at or above $500k
3. **Rental history** — has it ever been listed as a rental on Zillow/Redfin?
4. **Link to photos** — direct link to the Redfin or Zillow listing page with photos
5. **Last sold within 36 months** — recent sale = potential motivated seller signal
6. **Active or expired listing within last 36 months** — recent market activity

**Why:** These are the deal qualification criteria for their real estate investment operation. Leads that hit multiple flags = hot lead for closers.

**How to apply:** The Claude analysis prompt and Slack notification should be structured around these 6 flags specifically. MIN_VALUE_THRESHOLD should default to 500000.

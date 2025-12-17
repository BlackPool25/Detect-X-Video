# Fixing Modal Billing Limit Issue

## Problem
You're getting this error:
```
Error: Resource exhausted
App creation failed: workspace billing cycle spend limit reached
```

Even though you have $28 in credits remaining.

## Root Cause
Modal has two separate limits:
1. **Credits**: Your account balance ($28 remaining) ✅
2. **Billing Cycle Spend Limit**: A safety cap on how much you can spend per billing cycle ❌

You've hit the **spend limit**, not run out of credits.

## Solution

### Option 1: Increase Spend Limit (Recommended)

1. **Go to Modal Dashboard**: https://modal.com/settings
2. **Navigate to Billing**: Look for "Billing" or "Usage" tab
3. **Find Spend Limit**: Look for "Billing cycle spend limit" or similar
4. **Increase Limit**: Set it higher (e.g., $30-50 to use your credits)
5. **Save**: Apply the changes

### Option 2: Wait for Next Billing Cycle

If you don't want to increase the limit, wait until the billing cycle resets (usually monthly).

### Option 3: Contact Modal Support

If you can't find the setting:
1. Go to Modal Discord: https://discord.gg/modal
2. Or email: support@modal.com
3. Ask them to increase your spend limit

## Deploy After Fixing

Once the limit is increased, retry:

```bash
modal deploy modal_app_balanced.py
```

## Cost Optimization Changes Made

To reduce costs, I've already updated the code:
- Changed `keep_warm=1` to `min_containers=0` (no warm containers)
- This means cold starts (~10-15s first request) but $0 idle cost
- Subsequent requests are fast if within scale-down window

## Expected Costs

For reference, here's what the deployment will cost:

**Build Phase** (one-time per deployment):
- Docker image build: ~$0.10-0.20
- Weight upload (one-time): ~$0.05

**Runtime** (per request):
- T4 GPU: ~$0.000467/second
- 5 seconds per video: ~$0.0023 per detection
- 100 videos: ~$0.23
- 1000 videos: ~$2.30

**Idle Cost**: $0 (min_containers=0)

## Next Steps

1. Increase spend limit in Modal dashboard
2. Run `modal deploy modal_app_balanced.py`
3. Test with `python test_modal_balanced.py --url <your-url> --test-video <video-url>`

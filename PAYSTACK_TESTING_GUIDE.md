# Paystack Payment Gateway Testing Guide

## Setup Complete ✅

Your payment gateway has been configured with:
- **Plan:** Trading Dashboard Monthly
- **Plan Code:** `PLN_o6aocIukczuw4dk`
- **Amount:** ZAR 200.00
- **Test Secret Key:** Configured in `.env`
- **Test Public Key:** Configured in `.env`

---

## Prerequisites

1. ✅ Paystack account created
2. ✅ Test API keys obtained
3. ✅ Plan created in Paystack dashboard
4. ✅ Code updated with correct plan code
5. ✅ .env file updated with real keys
6. ⏳ Database migration (see below)
7. ⏳ Webhook configuration (see below)

---

## Step 1: Run Database Migration

```bash
cd "/home/calvin/Websites/Trading Dashboard/MoneyMoney"

# Connect to your PostgreSQL database and run migration
psql -U postgres -d trading_platform -f database/create_subscription_tables.sql
```

**What this creates:**
- Adds subscription fields to `users` table
- Creates `payment_history` table
- Creates `webhook_logs` table
- Creates `subscription_plans` table
- Inserts your ZAR 200 plan

**Expected output:**
```
NOTICE:  ✓ Subscription tables created successfully
NOTICE:  ✓ Default ZAR 200/month plan inserted
NOTICE:
NOTICE:  Next steps:
NOTICE:  1. Configure Paystack API keys in .env
NOTICE:  2. Plan already created in Paystack dashboard: PLN_o6aocIukczuw4dk
NOTICE:  3. Configure webhook URL in Paystack dashboard
NOTICE:  4. Test subscription flow
```

---

## Step 2: Setup Webhook Testing with ngrok

### Install ngrok (if not installed)
```bash
# Download from https://ngrok.com/download
# Or install via snap:
sudo snap install ngrok
```

### Start ngrok tunnel
```bash
ngrok http 3000
```

### Configure Webhook in Paystack
1. Copy the HTTPS URL from ngrok (e.g., `https://abc123.ngrok-free.app`)
2. Go to https://dashboard.paystack.com/#/settings/developer
3. Scroll to "Webhook URL"
4. Enter: `https://YOUR_NGROK_URL/api/payments/webhook`
5. Click "Save Changes"

**Example:** `https://abc123.ngrok-free.app/api/payments/webhook`

---

## Step 3: Start Your Server

```bash
cd "/home/calvin/Websites/Trading Dashboard/MoneyMoney"

# Kill any existing servers
pkill -f "node.*server.js"

# Start fresh
npm start
```

**Expected output:**
```
Server running on http://localhost:3000
Paystack keys loaded: sk_test_a057...
Database connected
```

---

## Step 4: Test Payment Flow

### A. Register Test User

1. Navigate to: http://localhost:3000/register
2. Register with test email: `test@example.com`
3. Password: `Test123!`
4. Login with credentials

### B. Access Subscription Page

1. Go to: http://localhost:3000/subscription
2. Should see:
   - ❌ Status: "Inactive"
   - 💰 Pricing: **R200/month**
   - 🔘 "Subscribe Now" button

### C. Initiate Payment

1. Click **"Subscribe Now"**
2. Paystack popup appears
3. Verify amount: **ZAR 200.00**

### D. Complete Test Payment

**Use Paystack Test Card:**
```
Card Number: 4084 0840 8408 4081
CVV: 408
Expiry: 12/26 (any future date)
PIN: 0000
OTP: 123456
```

**Steps:**
1. Enter card details
2. Click "Pay ZAR 200.00"
3. Enter PIN: `0000`
4. Enter OTP: `123456`
5. Payment processes

### E. Verify Success

1. Redirect to success page
2. Navigate to: http://localhost:3000/subscription
3. Should show:
   - ✅ Status: **"Active"**
   - 📅 Expires: **1 month from today**
   - 🔄 Auto-renew: **Enabled**
   - 📊 "Go to Dashboard" button

4. Test dashboard access: http://localhost:3000/dashboard
   - Should now work (previously blocked)

---

## Step 5: Verify Database Records

```bash
psql -U postgres -d trading_platform
```

### Check User Subscription
```sql
SELECT
    id,
    email,
    subscription_status,
    subscription_expires_at,
    auto_renew
FROM users
WHERE email = 'test@example.com';
```

**Expected:**
```
 id |       email        | subscription_status | subscription_expires_at | auto_renew
----+--------------------+---------------------+-------------------------+------------
  1 | test@example.com   | active              | 2025-12-27 10:30:00     | t
```

### Check Payment History
```sql
SELECT
    reference,
    amount,
    currency,
    status,
    paid_at
FROM payment_history
WHERE user_id = (SELECT id FROM users WHERE email = 'test@example.com')
ORDER BY created_at DESC;
```

**Expected:**
```
      reference      | amount | currency | status  |       paid_at
---------------------+--------+----------+---------+----------------------
 TRX_1701234567890   | 200.00 | ZAR      | success | 2025-11-27 10:30:15
```

### Check Webhook Logs
```sql
SELECT
    event_type,
    processed,
    received_at
FROM webhook_logs
ORDER BY received_at DESC
LIMIT 5;
```

**Expected:**
```
     event_type      | processed |     received_at
---------------------+-----------+----------------------
 charge.success      | t         | 2025-11-27 10:30:16
 subscription.create | t         | 2025-11-27 10:30:17
```

---

## Step 6: Test API Endpoints

### Get Subscription Status
```bash
# Get JWT token from browser localStorage
# Then test:

curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  http://localhost:3000/api/subscription/status | jq
```

**Expected response:**
```json
{
  "status": "active",
  "subscription_code": "SUB_xyz123",
  "started_at": "2025-11-27T10:30:00.000Z",
  "expires_at": "2025-12-27T10:30:00.000Z",
  "auto_renew": true,
  "days_remaining": 30
}
```

---

## Step 7: Test Subscription Cancellation

1. While logged in, go to: http://localhost:3000/subscription
2. Click **"Cancel Subscription"** button
3. Confirm cancellation

**Verify:**
- `auto_renew` changes to `false`
- Access continues until expiration date
- Shows "Will expire on [date]" message

### Database Check
```sql
SELECT auto_renew FROM users WHERE email = 'test@example.com';
```

**Expected:** `f` (false)

---

## Troubleshooting

### Payment Popup Doesn't Appear
- Check browser console for JavaScript errors
- Verify `PAYSTACK_PUBLIC_KEY` in .env is correct
- Check server logs for errors

### 404 Error on Payment
- Verify plan code: `PLN_o6aocIukczuw4dk`
- Check Paystack dashboard - plan should be "Active"
- Verify amount: `20000` cents (ZAR 200.00)

### Webhook Not Firing
- Verify ngrok is running: `ngrok http 3000`
- Check webhook URL in Paystack dashboard
- Must be HTTPS URL from ngrok
- Check webhook_logs table for errors

### Database Errors
- Ensure PostgreSQL is running: `sudo systemctl status postgresql`
- Verify database exists: `psql -l | grep trading_platform`
- Check connection string in .env

### "Subscription Not Active" After Payment
- Check payment_history table - was payment recorded?
- Check webhook_logs - were webhooks received?
- Verify webhook signature verification isn't failing
- Check server logs for errors

---

## Monitoring

### Watch Server Logs
```bash
# In terminal where server is running
# Watch for payment events
```

### Monitor Webhooks in Real-time
```bash
watch -n 2 "psql -U postgres -d trading_platform -c 'SELECT event_type, processed, received_at FROM webhook_logs ORDER BY received_at DESC LIMIT 5;'"
```

### Check Payment Success Rate
```sql
SELECT
    status,
    COUNT(*) as count,
    SUM(amount) as total_zar
FROM payment_history
GROUP BY status;
```

---

## Test Checklist

- [ ] Database migration completed successfully
- [ ] ngrok tunnel running and webhook URL configured
- [ ] Server starts without errors
- [ ] Can register new user
- [ ] Subscription page shows R200 pricing
- [ ] Payment popup displays ZAR 200.00
- [ ] Test card payment completes
- [ ] Subscription status updates to "active"
- [ ] Dashboard becomes accessible
- [ ] Payment recorded in `payment_history`
- [ ] Webhooks logged in `webhook_logs`
- [ ] Can cancel subscription
- [ ] Auto-renew flag updates correctly

---

## Going to Production

When ready for live payments:

1. **Switch to Live Keys:**
   ```env
   PAYSTACK_SECRET_KEY=sk_live_YOUR_LIVE_KEY
   PAYSTACK_PUBLIC_KEY=pk_live_YOUR_LIVE_KEY
   ```

2. **Create Plan in Live Mode:**
   - Create identical plan in Paystack dashboard (Live mode)
   - Copy new live plan code
   - Update code with live plan code

3. **Update Webhook URL:**
   - Use actual production domain (not ngrok)
   - Configure in Paystack Live dashboard

4. **Security Checklist:**
   - [ ] Change JWT_SECRET to strong random value
   - [ ] Set NODE_ENV=production
   - [ ] Enable HTTPS
   - [ ] Set specific CORS_ORIGIN
   - [ ] Review rate limits
   - [ ] Add logging/monitoring
   - [ ] Set up automated backups

---

## Support

**Paystack Documentation:** https://paystack.com/docs
**Test Cards:** https://paystack.com/docs/payments/test-cards
**Dashboard:** https://dashboard.paystack.com

**Questions?** Check server logs and database records first!

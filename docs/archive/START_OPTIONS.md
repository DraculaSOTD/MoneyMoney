# 🚀 How to Start the Services

You have **3 different ways** to start the Trading Dashboard services. Choose whichever you prefer!

---

## Option 1: Separate Windows (start-all.sh) ⭐ Recommended

Opens two separate terminal windows - one for each service.

```bash
./start-all.sh
```

**Supported terminals:**
- Kitty ⭐ (you have this!)
- Alacritty
- WezTerm
- Terminator
- Tilix
- GNOME Terminal
- Konsole
- XTerm

**Result:**
- Window 1: Trading Platform (Backend) on port 8001
- Window 2: MoneyMoney (Frontend) on port 3000

**Stop:**
- Close the terminal windows OR
- Press `Ctrl+C` in each window

---

## Option 2: Split Terminal (start-services.sh)

Runs both services in one terminal with split panes using **tmux** or **screen**.

```bash
./start-services.sh
```

**Requirements:**
- tmux (recommended): `sudo pacman -S tmux`
- OR screen (fallback): `sudo pacman -S screen`

**Result:**
- Left pane: Trading Platform
- Right pane: MoneyMoney
- Both visible in one window!

**Tmux Commands:**
- Attach: `tmux attach -t trading_dashboard`
- Detach: `Ctrl+B` then `D`
- Switch panes: `Ctrl+B` then arrow keys
- Stop: `tmux kill-session -t trading_dashboard`

**Screen Commands:**
- Attach: `screen -r trading_dashboard`
- Detach: `Ctrl+A` then `D`
- Switch windows: `Ctrl+A` then `N` (next) or `P` (previous)
- Stop: `screen -S trading_dashboard -X quit`

---

## Option 3: Manual (Two Terminals)

Open two terminal windows/tabs yourself and run:

**Terminal 1 - Trading Platform:**
```bash
cd "/home/calvin/Websites/Trading Dashboard/MoneyMoney/other platform/Trading"
./start.sh
```

**Terminal 2 - MoneyMoney:**
```bash
cd "/home/calvin/Websites/Trading Dashboard/MoneyMoney"
./start.sh
```

**Stop:**
- Press `Ctrl+C` in each terminal

---

## Quick Comparison

| Method | Pros | Cons |
|--------|------|------|
| **start-all.sh** | ✅ Automatic<br>✅ Separate windows<br>✅ Easy to monitor | ⚠️ Requires GUI terminal |
| **start-services.sh** | ✅ Single window<br>✅ Split view<br>✅ Tmux features | ⚠️ Requires tmux/screen<br>⚠️ Different key bindings |
| **Manual** | ✅ Full control<br>✅ No dependencies | ⚠️ Manual setup<br>⚠️ More steps |

---

## After Starting

Once services are running (any method), access:

- **MoneyMoney Login**: http://localhost:3000/auth
- **Dashboard**: http://localhost:3000/dashboard
- **API Docs**: http://localhost:8001/docs

**Login credentials:**
- Email: `test@example.com`
- Password: `testpassword`

---

## Troubleshooting

### "Port already in use"

```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Kill process on port 8001
lsof -ti:8001 | xargs kill -9
```

### "No terminal emulator found" (start-all.sh)

Install one:
```bash
sudo pacman -S kitty    # Recommended
# OR
sudo pacman -S alacritty
```

Or use `start-services.sh` with tmux instead:
```bash
sudo pacman -S tmux
./start-services.sh
```

### "tmux not found" (start-services.sh)

Install tmux:
```bash
sudo pacman -S tmux
```

Or use manual method (Option 3).

---

## Recommended for You

Since you have **Kitty terminal**, I recommend:

```bash
./start-all.sh
```

This will open two Kitty windows automatically. It's the easiest method!

---

**Happy trading! 💰📈**

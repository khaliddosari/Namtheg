# Frontend e2e checks

Browser checks for the dashboard, run against a mock of the backend API so they
need no Modal, GPU or R2:

- `suite.js`: 58 end-to-end flows (the intro slide, upload, configure and a
  column to add, training and its clock, results and paging through the
  graphs, export, deploy, predict with category lists, share page, sign-in
  and accounts, routing, navigation, loading and error states). The export
  flow measures the report as it is drawn (card order, the graph on screen,
  text cut off, empty space in a card), checks the PDF's A4 pages, and
  downloads the graphs' zip.
  Any console or page error fails a test, except the HTTP errors a test
  provokes on purpose (the R2 fallback's 409, a missing plot's 404, a failed
  predict's 502, a wrong sign-in code's 400, an upload read that a proxy
  drops).
- `layout.js`: the intro and every stage at seven viewports, from 1920×1080
  down to 360×740. It flags sideways page scroll, content escaping its stage
  or its control, text cut off without an ellipsis, and empty gaps under
  locked pages. Screenshots land in `shots/layout` for a look by eye.
- `a11y.js`: axe-core WCAG 2.1 A/AA on each screen at desktop and phone width,
  plus a keyboard walk that checks every focus stop shows a focus ring, then
  on each dialog: the rationale, the analyst's findings, the zoomed graph,
  cancelling a run, every sign-up step, and the account.

## Setup, once

```bash
cd Frontend/e2e
npm install
npm run install-browser   # skip if you set CHROME_PATH or already have a Playwright Chromium
```

The mock needs a Python with `fastapi` and `uvicorn`. The backend's virtualenv
has both.

## Run

Three terminals, all in `Frontend/e2e`:

```bash
npm run mock                     # 1. mock backend on :8001
npm run build && npm run serve   # 2. isolated build of the frontend on :3001
npm run check                    # 3. suite, then layout, then accessibility
```

`npm test`, `npm run layout` and `npm run a11y` run one of them. `npm test -- "deploy"`
runs only the tests whose name contains "deploy". A full `check` takes about seven
minutes; two tests wait on purpose: one for a minute, because the app retries a
missing result for a minute before it gives up, and one for half a minute, to
outlast a proxy that stops waiting on a slow upload read.

After changing frontend code, stop `serve`, then `npm run build` and `npm run serve`
again. `npm run clean` deletes the build copy when you are done.

## Why a copy

`npm run build` copies the frontend to your temp folder and builds it there
against the mock, with `node_modules` linked rather than copied. Running
`next build` inside `Frontend/` would overwrite the `.next` folder a running
`npm run dev` serves from and break it until restarted. `clean` removes the
`node_modules` link before it deletes the copy, so your real `node_modules` is
never touched.

## The link preview image

`app/opengraph-image.png` is the picture link previews show. `npm run og-image`
redraws it from the page in `og-image.js`, in the same browser, with the site's
fonts loaded from jsDelivr. A browser draws the wordmark's swash forms, which
Next's generated images can't. Redraw it after changing the brand or the theme.

## Seeded runs

The mock starts with these run ids, and `POST /__mock/reset` restores them.
They also work for browsing by hand, for example
`http://localhost:3001/?run=reg-done&stage=results`.

| Run id | Scenario |
| --- | --- |
| `fresh` | Uploaded, not started |
| `reading` | A large upload still being read on the server |
| `reg-done` | Regression with tuning, features, leaderboard, downloads |
| `clf-done` | Classification with an overfit gap |
| `bare-done` | A result with no extras and no plot |
| `running`, `queued` | Frozen mid-run |
| `failed`, `cancelled` | Stopped runs |
| `live-done`, `liveclf-done` | Already deployed (regression, classification) |
| `deployfail-done` | First deploy fails, retry succeeds |
| `predicterr-done` | Deployed, predictions fail with a 502 |
| `noresult-done` | Succeeded, but the result never loads |
| `schemaerr-done` | Succeeded, but the model details fail |

`POST /__mock/config` with `{"r2": true}` turns on the direct R2 upload path,
and `{"delay": 2}` slows every run read by two seconds to show loading states.
`/uploads/{id}/complete` answers the way the backend does: 202 at once, with the
file read in the background for `read_seconds` (1). `{"complete": ...}` picks
another way: `fail` (the read fails with a reason), the older backend that read
inside the request, which the frontend still handles (`dropped`: a plain 500
after 27 seconds, the way a proxy that stopped waiting answers, with the run
ready soon after; `reject`: a 400 with the reason), or `crash` (a plain 500 at
once). See `complete()` in `mock_backend.py`.

Accounts are on, as on a backend with email set up; `{"auth": false}` turns
them off, the way a backend without email answers. No email is sent:
`GET /__mock/last_code?email=...` returns the code that would have gone out,
and `POST /__mock/account` with `{"email", "name", "runs": [run ids]}` makes
an account that already lists runs, oldest first. `{"resend_seconds": 0}`
lifts the minute an address waits between codes.

## Settings

| Variable | Default | What it sets |
| --- | --- | --- |
| `E2E_APP_URL` | `http://localhost:3001` | Where the scripts open the app, and the port `serve` uses |
| `E2E_MOCK_URL` | `http://localhost:8001` | The mock the build proxies to and the scripts reset |
| `E2E_APP_DIR` | `<temp>/namtheg-e2e-app` | Where the build copy lives (must be outside `Frontend/`) |
| `CHROME_PATH` | a Playwright Chromium if found | The browser the scripts drive |

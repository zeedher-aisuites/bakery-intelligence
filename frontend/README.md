# Bakery Intelligence frontend

React 18 + TypeScript + Vite application. It provides the
landing page, the New Recipe upload and authorization flow, the
processing queue, the three-stage review workspace (Review
findings, Generate recipe, Review recipe), recipe version
history, and PDF/DOCX export download.

## Prerequisites

- Node.js 20 or later (`node --version`).
- The FastAPI backend running on `http://localhost:8000` (see
  the root README).

## Install and run (Windows PowerShell)

```powershell
npm install --prefix frontend
npm run dev --prefix frontend
```

Open the printed URL (default `http://localhost:5173`) in a browser.

To point at a different backend, create `frontend/.env` (gitignored):

```text
VITE_API_BASE=http://localhost:8000
```

## Test and build

```powershell
npm test --prefix frontend
npm run build --prefix frontend
```

`npm test` runs `node --test` over `frontend/tests/`, including
jsdom-rendered interaction tests for navigation, the empty
states, recipe families, and the review workflow. `npm run
build` runs `tsc --noEmit` followed by the production Vite
build into `frontend/dist/` (git-ignored).

## Source layout

- `src/api.ts`, `src/upload.ts`, `src/queue.ts`, `src/recipe.ts`,
  `src/review.ts` — typed API clients and pure helpers.
- `src/App.tsx` — routes and top-level navigation.
- `src/UploadWorkflow.tsx` — New Recipe upload, engine
  selection, authorization, and progress display.
- `src/ReviewWorkspace.tsx` — landing page, recipe library, and
  the staged review workspace.
- `src/RecipeWorkspace.tsx`, `src/RecipeReview.tsx`,
  `src/RecipeEditor.tsx` — recipe viewing, review, and editing.
- `src/ObservationReview.tsx`, `src/EvidenceExplorer.tsx`,
  `src/FrameViewer.tsx` — finding review and frame evidence.
- `src/GeneratePanel.tsx` — recipe generation readiness and
  execution.
- `src/*.css` — design tokens (`tokens.css`), base styles, and
  the premium editorial theme (`premium.css`).

All UI copy is in English; recipe content preserves the language
selected at creation time.

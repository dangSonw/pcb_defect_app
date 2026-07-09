"""
Tab 6 — Full AI Pipeline (Tree Diagram, light mode)
Large, high-contrast, fully animated tree layout with sequential steps.
"""
import gradio as gr

# ──────────────────────────────────────────────────────────────────────────
#  Self-contained HTML + CSS
# ──────────────────────────────────────────────────────────────────────────
_HTML = r"""
<style>
  /* ── reset ───────────────────────────────────────────────────────────── */
  .fp6 *, .fp6 *::before, .fp6 *::after { box-sizing: border-box; margin:0; padding:0; }

  /* ── outer card ──────────────────────────────────────────────────────── */
  .fp6 {
    width: 100%; max-width: 100%; overflow: hidden;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 24px;
    padding: 44px 38px 44px;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.05), 0 2px 4px rgba(0, 0, 0, 0.02);
  }

  /* ── page title ──────────────────────────────────────────────────────── */
  .fp6-title {
    text-align: center;
    font-size: 1.5rem; font-weight: 900;
    color: #0f172a; letter-spacing: -.020em;
    margin-bottom: 38px;
    animation: fp6-fadein .6s cubic-bezier(0.16, 1, 0.3, 1) both;
  }

  /* ── root node ───────────────────────────────────────────────────────── */
  .fp6-root-area {
    display: flex; flex-direction: column; align-items: center;
  }
  .fp6-root-box {
    position: relative;
    overflow: hidden;
    border-radius: 16px;
    padding: 18px 56px;
    font-weight: 900; font-size: 1.12rem;
    text-align: center;
    border: 3px solid #cbd5e1;
    box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    animation: fp6-loop-step0 11s infinite ease-in-out;
  }
  .fp6-root-box:hover {
    transform: translateY(-2px) scale(1.02);
  }
  .fp6-root-box::after {
    content: '';
    position: absolute; top: -50%; left: -60%;
    width: 30%; height: 200%;
    background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.75) 50%, rgba(255,255,255,0) 100%);
    transform: rotate(30deg);
    animation: fp6-sheen 3.5s infinite linear;
  }

  /* short vertical stub from root down */
  .fp6-root-stub {
    width: 3.5px; height: 30px;
    background: #cbd5e1;
    position: relative; overflow: hidden;
  }
  .fp6-root-stub::after {
    content: '';
    position: absolute; top: -100%; left: 0;
    width: 100%; height: 50%;
    background: linear-gradient(to bottom, transparent, #0ea5e9, transparent);
    animation: fp6-flow-v 2s linear infinite;
  }

  /* ── T-connector bar ─────────────────────────────────────────────────── */
  .fp6-tbar-wrap {
    width: 50%; margin: 0 auto;
    position: relative; height: 50px;
  }

  /* horizontal bar */
  .fp6-tbar-h {
    position: absolute; top: 0; left: 0; right: 0;
    height: 3.5px; overflow: hidden; background: #e2e8f0;
  }
  .fp6-tbar-h::after {
    content: '';
    position: absolute; top: 0; left: -80%;
    width: 60%; height: 100%;
    background: linear-gradient(to right, transparent, #38bdf8 50%, transparent);
    animation: fp6-flow-h 2.8s linear infinite;
  }

  /* left vertical drop  */
  .fp6-drop-l {
    position: absolute; top: 0; left: 0;
    width: 3.5px; height: 50px;
    background: #e2e8f0; overflow: hidden;
  }
  .fp6-drop-l-shine {
    position: absolute; top: -100%; left: 0;
    width: 100%; height: 50%;
    background: linear-gradient(to bottom, transparent, #3b82f6, transparent);
    animation: fp6-flow-v 1.8s linear infinite .3s;
  }
  .fp6-drop-l-head {
    position: absolute; bottom: -9px; left: -5.5px;
    width: 0; height: 0;
    border-left: 7px solid transparent;
    border-right: 7px solid transparent;
    border-top: 10px solid #3b82f6;
  }

  /* right vertical drop */
  .fp6-drop-r {
    position: absolute; top: 0; right: 0;
    width: 3.5px; height: 50px;
    background: #e2e8f0; overflow: hidden;
  }
  .fp6-drop-r-shine {
    position: absolute; top: -100%; left: 0;
    width: 100%; height: 50%;
    background: linear-gradient(to bottom, transparent, #10b981, transparent);
    animation: fp6-flow-v 1.8s linear infinite .7s;
  }
  .fp6-drop-r-head {
    position: absolute; bottom: -9px; left: -5.5px;
    width: 0; height: 0;
    border-left: 7px solid transparent;
    border-right: 7px solid transparent;
    border-top: 10px solid #10b981;
  }

  /* branch labels (Inference / Training) */
  .fp6-branch-lbl {
    position: absolute; top: -28px;
    font-weight: 900;
    font-size: 0.98rem; letter-spacing: -0.01em; white-space: nowrap;
    padding: 3px 10px;
    border-radius: 8px;
  }
  .fp6-branch-lbl-l {
    left: 10%; color: #2563eb; background: #eff6ff;
  }
  .fp6-branch-lbl-r {
    right: 10%; color: #059669; background: #ecfdf5;
  }

  /* ── two-column content grid ─────────────────────────────────────────── */
  .fp6-cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 56px;
    margin-top: 10px;
  }

  /* ── column ──────────────────────────────────────────────────────────── */
  .fp6-col {
    display: flex; flex-direction: column; align-items: center;
    width: 100%;
  }

  /* ── node group: [box | bullets] + connector ─────────────────────────── */
  .fp6-ng { width: 100%; }

  .fp6-node-row {
    display: grid;
    grid-template-columns: 0.9fr auto 1.2fr;
    align-items: center;
    gap: 24px;
    width: 100%;
  }

  /* node box base style */
  .fp6-nbox {
    position: relative;
    overflow: hidden;
    border-radius: 16px;
    padding: 18px 22px;
    font-weight: 900; font-size: 1.05rem;
    text-align: center;
    width: 160px;
    flex-shrink: 0;
    border: 3px solid #cbd5e1;
    background: #ffffff;
    box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    transition: transform 0.3s ease;
    cursor: default;
  }
  .fp6-nbox:hover {
    transform: translateY(-3px) scale(1.02);
  }
  .fp6-nbox::after {
    content: '';
    position: absolute; top: -50%; left: -60%;
    width: 35%; height: 200%;
    background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.75) 50%, rgba(255,255,255,0) 100%);
    transform: rotate(30deg);
    animation: fp6-sheen 3.2s infinite linear;
  }

  /* bullet list */
  .fp6-blist {
    display: flex; flex-direction: column; gap: 8px;
    padding-top: 2px; flex: 1; min-width: 0;
  }
  .fp6-b {
    display: flex; align-items: center; gap: 10px;
    font-size: 0.96rem; font-weight: 700; line-height: 1.4;
    transition: color 0.3s, font-weight 0.3s;
  }
  .fp6-b-dot {
    width: 11px; height: 11px;
    border-radius: 50%; border: 2.2px solid currentColor;
    flex-shrink: 0; display: inline-block;
    transition: transform 0.3s, background-color 0.3s;
  }

  /* vertical connector arrow between nodes */
  .fp6-vc {
    display: flex; flex-direction: column; align-items: center;
    width: 160px;
    padding: 6px 0;
  }
  .fp6-vc-line {
    width: 3.5px; height: 36px;
    position: relative; overflow: hidden;
    background: #cbd5e1;
  }
  .fp6-vc-shine {
    position: absolute; top: -100%; left: 0;
    width: 100%; height: 50%;
  }
  .fp6-vc-head {
    width: 0; height: 0;
    border-left: 7px solid transparent;
    border-right: 7px solid transparent;
    border-top: 10px solid #cbd5e1;
    margin-top: -1px;
  }

  /* ──────────────────────────────────────────────────────────────────────────
     SEQUENTIAL PIPELINE ANIMATION (11s loop)
     Steps:
       0% - 18%  : Root (Data Acquisition)
       20% - 38% : Step 1 (Pre-processing / Data Labeling)
       40% - 58% : Step 2 (Model Inference / Data Splitting)
       60% - 78% : Step 3 (Output / Model Training)
       80% - 98% : Step 4 (Model Optimization)
  ────────────────────────────────────────────────────────────────────────── */

  /* ── STEP 0 (Root) ── */
  @keyframes fp6-loop-step0 {
    0%, 18%, 100% {
      border-color: #0284c7; color: #0369a1;
      background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
      box-shadow: 0 8px 24px rgba(2, 132, 199, 0.25);
    }
    20%, 98% {
      border-color: #e2e8f0; color: #64748b; background: #ffffff;
      box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    }
  }

  /* ── STEP 1 (Pre-processing / Data Labeling) ── */
  .fp6-step1-inf-box { animation: fp6-loop-step1-inf-box 11s infinite ease-in-out; }
  .fp6-step1-train-box { animation: fp6-loop-step1-train-box 11s infinite ease-in-out; }

  @keyframes fp6-loop-step1-inf-box {
    20%, 38% {
      border-color: #3b82f6; color: #1e40af;
      background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
      box-shadow: 0 8px 24px rgba(59, 130, 246, 0.25);
    }
    0%, 18%, 40%, 100% {
      border-color: #e2e8f0; color: #64748b; background: #ffffff;
      box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    }
  }
  @keyframes fp6-loop-step1-train-box {
    20%, 38% {
      border-color: #10b981; color: #065f46;
      background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
      box-shadow: 0 8px 24px rgba(16, 185, 129, 0.25);
    }
    0%, 18%, 40%, 100% {
      border-color: #e2e8f0; color: #64748b; background: #ffffff;
      box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    }
  }

  /* Step 1 Bullets */
  .fp6-step1-b1 { animation: fp6-bullet-step1-b1 11s infinite ease-in-out; }
  .fp6-step1-b2 { animation: fp6-bullet-step1-b2 11s infinite ease-in-out; }
  .fp6-step1-b3 { animation: fp6-bullet-step1-b3 11s infinite ease-in-out; }

  @keyframes fp6-bullet-step1-b1 {
    20%, 38% { font-weight: 900; }
    20%, 38% { color: inherit; }
    0%, 18%, 40%, 100% { color: #94a3b8; font-weight: 600; }
  }
  @keyframes fp6-bullet-step1-b2 {
    26%, 38% { font-weight: 900; }
    26%, 38% { color: inherit; }
    0%, 24%, 40%, 100% { color: #94a3b8; font-weight: 600; }
  }
  @keyframes fp6-bullet-step1-b3 {
    32%, 38% { font-weight: 900; }
    32%, 38% { color: inherit; }
    0%, 30%, 40%, 100% { color: #94a3b8; font-weight: 600; }
  }

  /* Step 1 Dots */
  .fp6-step1-d1 { animation: fp6-dot-step1-d1 11s infinite ease-in-out; }
  .fp6-step1-d2 { animation: fp6-dot-step1-d2 11s infinite ease-in-out; }
  .fp6-step1-d3 { animation: fp6-dot-step1-d3 11s infinite ease-in-out; }

  @keyframes fp6-dot-step1-d1 {
    20%, 38% { background-color: currentColor; transform: scale(1.2); }
    0%, 18%, 40%, 100% { background-color: transparent; transform: scale(1); }
  }
  @keyframes fp6-dot-step1-d2 {
    26%, 38% { background-color: currentColor; transform: scale(1.2); }
    0%, 24%, 40%, 100% { background-color: transparent; transform: scale(1); }
  }
  @keyframes fp6-dot-step1-d3 {
    32%, 38% { background-color: currentColor; transform: scale(1.2); }
    0%, 30%, 40%, 100% { background-color: transparent; transform: scale(1); }
  }

  /* Step 1 connector line underneath */
  .fp6-step1-vc-line { animation: fp6-line-step1 11s infinite ease-in-out; }
  .fp6-step1-vc-shine { animation: fp6-flow-v 1.6s linear infinite; }
  @keyframes fp6-line-step1 {
    30%, 42% { background: #bfdbfe; }
    0%, 28%, 44%, 100% { background: #cbd5e1; }
  }
  .fp6-step1-vc-head { animation: fp6-head-step1 11s infinite ease-in-out; }
  @keyframes fp6-head-step1 {
    30%, 42% { border-top-color: currentColor; }
    0%, 28%, 44%, 100% { border-top-color: #cbd5e1; }
  }

  /* ── STEP 2 (Model Inference / Data Splitting) ── */
  .fp6-step2-inf-box { animation: fp6-loop-step2-inf-box 11s infinite ease-in-out; }
  .fp6-step2-train-box { animation: fp6-loop-step2-train-box 11s infinite ease-in-out; }

  @keyframes fp6-loop-step2-inf-box {
    40%, 58% {
      border-color: #3b82f6; color: #1e40af;
      background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
      box-shadow: 0 8px 24px rgba(59, 130, 246, 0.25);
    }
    0%, 38%, 60%, 100% {
      border-color: #e2e8f0; color: #64748b; background: #ffffff;
      box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    }
  }
  @keyframes fp6-loop-step2-train-box {
    40%, 58% {
      border-color: #10b981; color: #065f46;
      background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
      box-shadow: 0 8px 24px rgba(16, 185, 129, 0.25);
    }
    0%, 38%, 60%, 100% {
      border-color: #e2e8f0; color: #64748b; background: #ffffff;
      box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    }
  }

  /* Step 2 Bullets */
  .fp6-step2-b1 { animation: fp6-bullet-step2-b1 11s infinite ease-in-out; }
  .fp6-step2-b2 { animation: fp6-bullet-step2-b2 11s infinite ease-in-out; }
  .fp6-step2-b3 { animation: fp6-bullet-step2-b3 11s infinite ease-in-out; }

  @keyframes fp6-bullet-step2-b1 {
    40%, 58% { font-weight: 900; }
    40%, 58% { color: inherit; }
    0%, 38%, 60%, 100% { color: #94a3b8; font-weight: 600; }
  }
  @keyframes fp6-bullet-step2-b2 {
    46%, 58% { font-weight: 900; }
    46%, 58% { color: inherit; }
    0%, 44%, 60%, 100% { color: #94a3b8; font-weight: 600; }
  }
  @keyframes fp6-bullet-step2-b3 {
    52%, 58% { font-weight: 900; }
    52%, 58% { color: inherit; }
    0%, 50%, 60%, 100% { color: #94a3b8; font-weight: 600; }
  }

  /* Step 2 Dots */
  .fp6-step2-d1 { animation: fp6-dot-step2-d1 11s infinite ease-in-out; }
  .fp6-step2-d2 { animation: fp6-dot-step2-d2 11s infinite ease-in-out; }
  .fp6-step2-d3 { animation: fp6-dot-step2-d3 11s infinite ease-in-out; }

  @keyframes fp6-dot-step2-d1 {
    40%, 58% { background-color: currentColor; transform: scale(1.2); }
    0%, 38%, 60%, 100% { background-color: transparent; transform: scale(1); }
  }
  @keyframes fp6-dot-step2-d2 {
    46%, 58% { background-color: currentColor; transform: scale(1.2); }
    0%, 44%, 60%, 100% { background-color: transparent; transform: scale(1); }
  }
  @keyframes fp6-dot-step2-d3 {
    52%, 58% { background-color: currentColor; transform: scale(1.2); }
    0%, 50%, 60%, 100% { background-color: transparent; transform: scale(1); }
  }

  /* Step 2 connector line underneath */
  .fp6-step2-vc-line { animation: fp6-line-step2 11s infinite ease-in-out; }
  .fp6-step2-vc-shine { animation: fp6-flow-v 1.6s linear infinite; }
  @keyframes fp6-line-step2 {
    50%, 62% { background: #bfdbfe; }
    0%, 48%, 64%, 100% { background: #cbd5e1; }
  }
  .fp6-step2-vc-head { animation: fp6-head-step2 11s infinite ease-in-out; }
  @keyframes fp6-head-step2 {
    50%, 62% { border-top-color: currentColor; }
    0%, 48%, 64%, 100% { border-top-color: #cbd5e1; }
  }


  /* ── STEP 3 (Output / Model Training) ── */
  .fp6-step3-inf-box { animation: fp6-loop-step3-inf-box 11s infinite ease-in-out; }
  .fp6-step3-train-box { animation: fp6-loop-step3-train-box 11s infinite ease-in-out; }

  @keyframes fp6-loop-step3-inf-box {
    60%, 78% {
      border-color: #3b82f6; color: #1e40af;
      background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
      box-shadow: 0 8px 24px rgba(59, 130, 246, 0.25);
    }
    0%, 58%, 80%, 100% {
      border-color: #e2e8f0; color: #64748b; background: #ffffff;
      box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    }
  }
  @keyframes fp6-loop-step3-train-box {
    60%, 78% {
      border-color: #10b981; color: #065f46;
      background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
      box-shadow: 0 8px 24px rgba(16, 185, 129, 0.25);
    }
    0%, 58%, 80%, 100% {
      border-color: #e2e8f0; color: #64748b; background: #ffffff;
      box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    }
  }

  /* Step 3 Bullets (Inference has 2, Training has 4) */
  .fp6-step3-b1 { animation: fp6-bullet-step3-b1 11s infinite ease-in-out; }
  .fp6-step3-b2 { animation: fp6-bullet-step3-b2 11s infinite ease-in-out; }
  .fp6-step3-b3 { animation: fp6-bullet-step3-b3 11s infinite ease-in-out; }
  .fp6-step3-b4 { animation: fp6-bullet-step3-b4 11s infinite ease-in-out; }

  @keyframes fp6-bullet-step3-b1 {
    60%, 78% { font-weight: 900; }
    60%, 78% { color: inherit; }
    0%, 58%, 80%, 100% { color: #94a3b8; font-weight: 600; }
  }
  @keyframes fp6-bullet-step3-b2 {
    65%, 78% { font-weight: 900; }
    65%, 78% { color: inherit; }
    0%, 63%, 80%, 100% { color: #94a3b8; font-weight: 600; }
  }
  @keyframes fp6-bullet-step3-b3 {
    70%, 78% { font-weight: 900; }
    70%, 78% { color: inherit; }
    0%, 68%, 80%, 100% { color: #94a3b8; font-weight: 600; }
  }
  @keyframes fp6-bullet-step3-b4 {
    75%, 78% { font-weight: 900; }
    75%, 78% { color: inherit; }
    0%, 73%, 80%, 100% { color: #94a3b8; font-weight: 600; }
  }

  /* Step 3 Dots */
  .fp6-step3-d1 { animation: fp6-dot-step3-d1 11s infinite ease-in-out; }
  .fp6-step3-d2 { animation: fp6-dot-step3-d2 11s infinite ease-in-out; }
  .fp6-step3-d3 { animation: fp6-dot-step3-d3 11s infinite ease-in-out; }
  .fp6-step3-d4 { animation: fp6-dot-step3-d4 11s infinite ease-in-out; }

  @keyframes fp6-dot-step3-d1 {
    60%, 78% { background-color: currentColor; transform: scale(1.2); }
    0%, 58%, 80%, 100% { background-color: transparent; transform: scale(1); }
  }
  @keyframes fp6-dot-step3-d2 {
    65%, 78% { background-color: currentColor; transform: scale(1.2); }
    0%, 63%, 80%, 100% { background-color: transparent; transform: scale(1); }
  }
  @keyframes fp6-dot-step3-d3 {
    70%, 78% { background-color: currentColor; transform: scale(1.2); }
    0%, 68%, 80%, 100% { background-color: transparent; transform: scale(1); }
  }
  @keyframes fp6-dot-step3-d4 {
    75%, 78% { background-color: currentColor; transform: scale(1.2); }
    0%, 73%, 80%, 100% { background-color: transparent; transform: scale(1); }
  }

  /* Step 3 connector line underneath (only on Training branch) */
  .fp6-step3-vc-line { animation: fp6-line-step3 11s infinite ease-in-out; }
  @keyframes fp6-line-step3 {
    70%, 82% { background: #a7f3d0; }
    0%, 68%, 84%, 100% { background: #cbd5e1; }
  }
  .fp6-step3-vc-head { animation: fp6-head-step3 11s infinite ease-in-out; }
  @keyframes fp6-head-step3 {
    70%, 82% { border-top-color: currentColor; }
    0%, 68%, 84%, 100% { border-top-color: #cbd5e1; }
  }


  /* ── STEP 4 (Model Optimization - Right branch only) ── */
  .fp6-step4-box { animation: fp6-loop-step4-box 11s infinite ease-in-out; }
  @keyframes fp6-loop-step4-box {
    80%, 98% {
      border-color: #10b981; color: #065f46;
      background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
      box-shadow: 0 8px 24px rgba(16, 185, 129, 0.25);
    }
    0%, 78%, 100% {
      border-color: #e2e8f0; color: #64748b; background: #ffffff;
      box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    }
  }

  /* Step 4 Bullets */
  .fp6-step4-b1 { animation: fp6-bullet-step4-b1 11s infinite ease-in-out; }
  .fp6-step4-b2 { animation: fp6-bullet-step4-b2 11s infinite ease-in-out; }
  .fp6-step4-b3 { animation: fp6-bullet-step4-b3 11s infinite ease-in-out; }

  @keyframes fp6-bullet-step4-b1 {
    80%, 98% { font-weight: 900; }
    80%, 98% { color: inherit; }
    0%, 78%, 100% { color: #94a3b8; font-weight: 600; }
  }
  @keyframes fp6-bullet-step4-b2 {
    86%, 98% { font-weight: 900; }
    86%, 98% { color: inherit; }
    0%, 84%, 100% { color: #94a3b8; font-weight: 600; }
  }
  @keyframes fp6-bullet-step4-b3 {
    92%, 98% { font-weight: 900; }
    92%, 98% { color: inherit; }
    0%, 90%, 100% { color: #94a3b8; font-weight: 600; }
  }

  /* Step 4 Dots */
  .fp6-step4-d1 { animation: fp6-dot-step4-d1 11s infinite ease-in-out; }
  .fp6-step4-d2 { animation: fp6-dot-step4-d2 11s infinite ease-in-out; }
  .fp6-step4-d3 { animation: fp6-dot-step4-d3 11s infinite ease-in-out; }

  @keyframes fp6-dot-step4-d1 {
    80%, 98% { background-color: currentColor; transform: scale(1.2); }
    0%, 78%, 100% { background-color: transparent; transform: scale(1); }
  }
  @keyframes fp6-dot-step4-d2 {
    86%, 98% { background-color: currentColor; transform: scale(1.2); }
    0%, 84%, 100% { background-color: transparent; transform: scale(1); }
  }
  @keyframes fp6-dot-step4-d3 {
    92%, 98% { background-color: currentColor; transform: scale(1.2); }
    0%, 90%, 100% { background-color: transparent; transform: scale(1); }
  }


  /* ── keyframes ───────────────────────────────────────────────────────── */
  @keyframes fp6-fadein {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes fp6-slidein {
    from { opacity: 0; transform: translateX(-8px); }
    to   { opacity: 1; transform: translateX(0); }
  }
  @keyframes fp6-flow-v {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(250%); }
  }
  @keyframes fp6-flow-h {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(250%); }
  }
  @keyframes fp6-sheen {
    0% { left: -75%; }
    100% { left: 125%; }
  }
</style>

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<div class="fp6">
  <div class="fp6-title">🏭 AI Pipeline — PCB Defect Detection System</div>

  <!-- ROOT NODE -->
  <div class="fp6-root-area">
    <div class="fp6-root-box">Data Acquisition</div>
    <div class="fp6-root-stub"></div>
  </div>

  <!-- T-CONNECTOR: horizontal bar + two drops + labels -->
  <div class="fp6-tbar-wrap">
    <div class="fp6-tbar-h"></div>

    <div class="fp6-drop-l">
      <div class="fp6-drop-l-shine"></div>
      <div class="fp6-drop-l-head"></div>
    </div>
    <div class="fp6-drop-r">
      <div class="fp6-drop-r-shine"></div>
      <div class="fp6-drop-r-head"></div>
    </div>

    <span class="fp6-branch-lbl fp6-branch-lbl-l">Inference</span>
    <span class="fp6-branch-lbl fp6-branch-lbl-r">Training</span>
  </div>

  <!-- TWO CONTENT COLUMNS -->
  <div class="fp6-cols">

    <!-- ══ LEFT: INFERENCE (blue) ══════════════════════════════════════ -->
    <div class="fp6-col fp6-inf">

      <!-- Pre-processing -->
      <div class="fp6-ng">
        <div class="fp6-node-row">
          <div style="min-width: 0;"></div>
          <div class="fp6-nbox fp6-step1-inf-box">Pre-<br>processing</div>
          <div class="fp6-blist">
            <div class="fp6-b fp6-step1-b1"><span class="fp6-b-dot fp6-step1-d1"></span>Letterboxing &amp; Resizing</div>
            <div class="fp6-b fp6-step1-b2"><span class="fp6-b-dot fp6-step1-d2"></span>Color Space Conversion</div>
            <div class="fp6-b fp6-step1-b3"><span class="fp6-b-dot fp6-step1-d3"></span>Normalization &amp; Permutation</div>
          </div>
        </div>
      </div>
      <div class="fp6-vc"><div class="fp6-vc-line fp6-step1-vc-line"><div class="fp6-vc-shine fp6-step1-vc-shine"></div></div><div class="fp6-vc-head fp6-step1-vc-head"></div></div>

      <!-- Model Inference -->
      <div class="fp6-ng">
        <div class="fp6-node-row">
          <div style="min-width: 0;"></div>
          <div class="fp6-nbox fp6-step2-inf-box">Model<br>Inference</div>
          <div class="fp6-blist">
            <div class="fp6-b fp6-step2-b1"><span class="fp6-b-dot fp6-step2-d1"></span>Tensor Injection</div>
            <div class="fp6-b fp6-step2-b2"><span class="fp6-b-dot fp6-step2-d2"></span>Forward Propagation</div>
            <div class="fp6-b fp6-step2-b3"><span class="fp6-b-dot fp6-step2-d3"></span>Predicts Generation</div>
          </div>
        </div>
      </div>
      <div class="fp6-vc"><div class="fp6-vc-line fp6-step2-vc-line"><div class="fp6-vc-shine fp6-step2-vc-shine"></div></div><div class="fp6-vc-head fp6-step2-vc-head"></div></div>

      <!-- Output -->
      <div class="fp6-ng">
        <div class="fp6-node-row">
          <div style="min-width: 0;"></div>
          <div class="fp6-nbox fp6-step3-inf-box">Output</div>
          <div class="fp6-blist">
            <div class="fp6-b fp6-step3-b1"><span class="fp6-b-dot fp6-step3-d1"></span>UI Rendering</div>
            <div class="fp6-b fp6-step3-b2"><span class="fp6-b-dot fp6-step3-d2"></span>Database Logging</div>
          </div>
        </div>
      </div>

    </div><!-- end Inference col -->

    <!-- ══ RIGHT: TRAINING (green) ═════════════════════════════════════ -->
    <div class="fp6-col fp6-train">

      <!-- Data Labeling -->
      <div class="fp6-ng">
        <div class="fp6-node-row">
          <div style="min-width: 0;"></div>
          <div class="fp6-nbox fp6-step1-train-box">Data<br>Labeling</div>
          <div class="fp6-blist">
            <div class="fp6-b fp6-step1-b1"><span class="fp6-b-dot fp6-step1-d1"></span>Defect Region Bounding</div>
            <div class="fp6-b fp6-step1-b2"><span class="fp6-b-dot fp6-step1-d2"></span>Class Assignment</div>
            <div class="fp6-b fp6-step1-b3"><span class="fp6-b-dot fp6-step1-d3"></span>Exporting Manifest</div>
          </div>
        </div>
      </div>
      <div class="fp6-vc"><div class="fp6-vc-line fp6-step1-vc-line"><div class="fp6-vc-shine fp6-step1-vc-shine"></div></div><div class="fp6-vc-head fp6-step1-vc-head"></div></div>

      <!-- Data Splitting -->
      <div class="fp6-ng">
        <div class="fp6-node-row">
          <div style="min-width: 0;"></div>
          <div class="fp6-nbox fp6-step2-train-box">Data<br>Splitting</div>
          <div class="fp6-blist">
            <div class="fp6-b fp6-step2-b1"><span class="fp6-b-dot fp6-step2-d1"></span>Dataset Shuffling</div>
            <div class="fp6-b fp6-step2-b2"><span class="fp6-b-dot fp6-step2-d2"></span>Subset Generation</div>
            <div class="fp6-b fp6-step2-b3"><span class="fp6-b-dot fp6-step2-d3"></span>Configuration Mapping</div>
          </div>
        </div>
      </div>
      <div class="fp6-vc"><div class="fp6-vc-line fp6-step2-vc-line"><div class="fp6-vc-shine fp6-step2-vc-shine"></div></div><div class="fp6-vc-head fp6-step2-vc-head"></div></div>

      <!-- Model Training -->
      <div class="fp6-ng">
        <div class="fp6-node-row">
          <div style="min-width: 0;"></div>
          <div class="fp6-nbox fp6-step3-train-box">Model<br>Training</div>
          <div class="fp6-blist">
            <div class="fp6-b fp6-step3-b1"><span class="fp6-b-dot fp6-step3-d1"></span>Parameters Configuration</div>
            <div class="fp6-b fp6-step3-b2"><span class="fp6-b-dot fp6-step3-d2"></span>Data Augmentation</div>
            <div class="fp6-b fp6-step3-b3"><span class="fp6-b-dot fp6-step3-d3"></span>Forward Pass &amp; Loss Calculation</div>
            <div class="fp6-b fp6-step3-b4"><span class="fp6-b-dot fp6-step3-d4"></span>Backward Pass &amp; Weight Update</div>
          </div>
        </div>
      </div>
      <div class="fp6-vc"><div class="fp6-vc-line fp6-step3-vc-line"><div class="fp6-vc-shine fp6-step3-vc-shine"></div></div><div class="fp6-vc-head fp6-step3-vc-head"></div></div>

      <!-- Model Optimization -->
      <div class="fp6-ng">
        <div class="fp6-node-row">
          <div style="min-width: 0;"></div>
          <div class="fp6-nbox fp6-step4-box">Model<br>Optimization</div>
          <div class="fp6-blist">
            <div class="fp6-b fp6-step4-b1"><span class="fp6-b-dot fp6-step4-d1"></span>Export ONNX</div>
            <div class="fp6-b fp6-step4-b2"><span class="fp6-b-dot fp6-step4-d2"></span>TensorRT Compilation</div>
            <div class="fp6-b fp6-step4-b3"><span class="fp6-b-dot fp6-step4-d3"></span>Half-Precision Quantization</div>
          </div>
        </div>
      </div>

    </div><!-- end Training col -->

  </div><!-- end fp6-cols -->
</div><!-- end fp6 -->
"""


def render():
    with gr.Column():
        gr.HTML(value=_HTML)

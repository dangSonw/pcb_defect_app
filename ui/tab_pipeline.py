"""
Tab 6 — Full AI Pipeline (Inference-only, light mode)
Large, high-contrast, fully animated tree layout with sequential steps.
After Output, a feedback loop arrow returns to Data Acquisition.
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
    width: 100%; max-width: 100%; overflow: visible;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 24px;
    padding: 44px 80px 44px 38px;
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

  /* ── single-column centered layout ──────────────────────────────────── */
  .fp6-pipeline {
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
    max-width: 560px;
    margin: 0 auto;
  }

  /* ── root node ───────────────────────────────────────────────────────── */
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
    z-index: 2;
    background: #ffffff;
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

  /* ── node group: [box | bullets] + connector ─────────────────────────── */
  .fp6-ng { width: 100%; max-width: 560px; }

  .fp6-node-row {
    display: grid;
    grid-template-columns: auto 1.4fr;
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
    z-index: 2;
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

  /* ── Feedback loop path (Output → Data Acquisition) ─────────────────── */
  /* C-shaped path on the right side using CSS borders */
  .fp6-loop-path {
    position: absolute;
    top: 20px;
    bottom: 20px;
    right: -45px;
    width: 30px;
    border: 3px solid #93c5fd;
    border-left: none;
    border-radius: 0 14px 14px 0;
    pointer-events: none;
    z-index: 1;
  }
  /* animated shine traveling up the path */
  .fp6-loop-path::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    border: 3px solid transparent;
    border-left: none;
    border-radius: 0 14px 14px 0;
    border-right-color: #3b82f6;
    animation: fp6-loop-trace 3s linear infinite;
    opacity: 0.6;
  }
  /* arrowhead at the top-left of the loop, pointing left into Data Acquisition */
  .fp6-loop-arrowhead {
    position: absolute;
    top: -7px; left: -8px;
    width: 0; height: 0;
    border-top: 7px solid transparent;
    border-bottom: 7px solid transparent;
    border-right: 10px solid #3b82f6;
    z-index: 2;
  }
  /* label */
  .fp6-loop-label {
    position: absolute;
    top: 50%; right: 6px;
    transform: translateY(-50%) rotate(90deg);
    font-size: 0.68rem; font-weight: 800;
    color: #3b82f6; letter-spacing: 0.08em;
    white-space: nowrap;
    opacity: 0.55;
  }

  @keyframes fp6-loop-trace {
    0%   { clip-path: inset(100% 0 0 0); }
    50%  { clip-path: inset(0 0 0 0); }
    100% { clip-path: inset(0 0 100% 0); }
  }


  /* ──────────────────────────────────────────────────────────────────────────
     SEQUENTIAL PIPELINE ANIMATION (11s loop)
     Steps:
       0% - 22%  : Root (Data Acquisition)
       24% - 46% : Step 1 (Pre-processing)
       48% - 70% : Step 2 (Model Inference)
       72% - 94% : Step 3 (Output)
  ────────────────────────────────────────────────────────────────────────── */

  /* ── STEP 0 (Root) ── */
  @keyframes fp6-loop-step0 {
    0%, 22%, 100% {
      border-color: #0284c7; color: #0369a1;
      background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
      box-shadow: 0 8px 24px rgba(2, 132, 199, 0.25);
    }
    24%, 94% {
      border-color: #e2e8f0; color: #64748b; background: #ffffff;
      box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    }
  }

  /* ── STEP 1 (Pre-processing) ── */
  .fp6-step1-box { animation: fp6-loop-step1-box 11s infinite ease-in-out; }
  @keyframes fp6-loop-step1-box {
    24%, 46% {
      border-color: #3b82f6; color: #1e40af;
      background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
      box-shadow: 0 8px 24px rgba(59, 130, 246, 0.25);
    }
    0%, 22%, 48%, 100% {
      border-color: #e2e8f0; color: #64748b; background: #ffffff;
      box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    }
  }

  /* Step 1 Bullets */
  .fp6-step1-b1 { animation: fp6-bullet-step1-b1 11s infinite ease-in-out; }
  .fp6-step1-b2 { animation: fp6-bullet-step1-b2 11s infinite ease-in-out; }
  .fp6-step1-b3 { animation: fp6-bullet-step1-b3 11s infinite ease-in-out; }

  @keyframes fp6-bullet-step1-b1 {
    24%, 46% { font-weight: 900; color: inherit; }
    0%, 22%, 48%, 100% { color: #94a3b8; font-weight: 600; }
  }
  @keyframes fp6-bullet-step1-b2 {
    30%, 46% { font-weight: 900; color: inherit; }
    0%, 28%, 48%, 100% { color: #94a3b8; font-weight: 600; }
  }
  @keyframes fp6-bullet-step1-b3 {
    36%, 46% { font-weight: 900; color: inherit; }
    0%, 34%, 48%, 100% { color: #94a3b8; font-weight: 600; }
  }

  /* Step 1 Dots */
  .fp6-step1-d1 { animation: fp6-dot-step1-d1 11s infinite ease-in-out; }
  .fp6-step1-d2 { animation: fp6-dot-step1-d2 11s infinite ease-in-out; }
  .fp6-step1-d3 { animation: fp6-dot-step1-d3 11s infinite ease-in-out; }

  @keyframes fp6-dot-step1-d1 {
    24%, 46% { background-color: currentColor; transform: scale(1.2); }
    0%, 22%, 48%, 100% { background-color: transparent; transform: scale(1); }
  }
  @keyframes fp6-dot-step1-d2 {
    30%, 46% { background-color: currentColor; transform: scale(1.2); }
    0%, 28%, 48%, 100% { background-color: transparent; transform: scale(1); }
  }
  @keyframes fp6-dot-step1-d3 {
    36%, 46% { background-color: currentColor; transform: scale(1.2); }
    0%, 34%, 48%, 100% { background-color: transparent; transform: scale(1); }
  }

  /* Step 1 connector */
  .fp6-step1-vc-line { animation: fp6-line-step1 11s infinite ease-in-out; }
  .fp6-step1-vc-shine { animation: fp6-flow-v 1.6s linear infinite; background: linear-gradient(to bottom, transparent, #3b82f6, transparent); }
  @keyframes fp6-line-step1 {
    38%, 50% { background: #bfdbfe; }
    0%, 36%, 52%, 100% { background: #cbd5e1; }
  }
  .fp6-step1-vc-head { animation: fp6-head-step1 11s infinite ease-in-out; }
  @keyframes fp6-head-step1 {
    38%, 50% { border-top-color: #3b82f6; }
    0%, 36%, 52%, 100% { border-top-color: #cbd5e1; }
  }


  /* ── STEP 2 (Model Inference) ── */
  .fp6-step2-box { animation: fp6-loop-step2-box 11s infinite ease-in-out; }
  @keyframes fp6-loop-step2-box {
    48%, 70% {
      border-color: #3b82f6; color: #1e40af;
      background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
      box-shadow: 0 8px 24px rgba(59, 130, 246, 0.25);
    }
    0%, 46%, 72%, 100% {
      border-color: #e2e8f0; color: #64748b; background: #ffffff;
      box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    }
  }

  /* Step 2 Bullets */
  .fp6-step2-b1 { animation: fp6-bullet-step2-b1 11s infinite ease-in-out; }
  .fp6-step2-b2 { animation: fp6-bullet-step2-b2 11s infinite ease-in-out; }
  .fp6-step2-b3 { animation: fp6-bullet-step2-b3 11s infinite ease-in-out; }

  @keyframes fp6-bullet-step2-b1 {
    48%, 70% { font-weight: 900; color: inherit; }
    0%, 46%, 72%, 100% { color: #94a3b8; font-weight: 600; }
  }
  @keyframes fp6-bullet-step2-b2 {
    54%, 70% { font-weight: 900; color: inherit; }
    0%, 52%, 72%, 100% { color: #94a3b8; font-weight: 600; }
  }
  @keyframes fp6-bullet-step2-b3 {
    60%, 70% { font-weight: 900; color: inherit; }
    0%, 58%, 72%, 100% { color: #94a3b8; font-weight: 600; }
  }

  /* Step 2 Dots */
  .fp6-step2-d1 { animation: fp6-dot-step2-d1 11s infinite ease-in-out; }
  .fp6-step2-d2 { animation: fp6-dot-step2-d2 11s infinite ease-in-out; }
  .fp6-step2-d3 { animation: fp6-dot-step2-d3 11s infinite ease-in-out; }

  @keyframes fp6-dot-step2-d1 {
    48%, 70% { background-color: currentColor; transform: scale(1.2); }
    0%, 46%, 72%, 100% { background-color: transparent; transform: scale(1); }
  }
  @keyframes fp6-dot-step2-d2 {
    54%, 70% { background-color: currentColor; transform: scale(1.2); }
    0%, 52%, 72%, 100% { background-color: transparent; transform: scale(1); }
  }
  @keyframes fp6-dot-step2-d3 {
    60%, 70% { background-color: currentColor; transform: scale(1.2); }
    0%, 58%, 72%, 100% { background-color: transparent; transform: scale(1); }
  }

  /* Step 2 connector */
  .fp6-step2-vc-line { animation: fp6-line-step2 11s infinite ease-in-out; }
  .fp6-step2-vc-shine { animation: fp6-flow-v 1.6s linear infinite; background: linear-gradient(to bottom, transparent, #3b82f6, transparent); }
  @keyframes fp6-line-step2 {
    62%, 74% { background: #bfdbfe; }
    0%, 60%, 76%, 100% { background: #cbd5e1; }
  }
  .fp6-step2-vc-head { animation: fp6-head-step2 11s infinite ease-in-out; }
  @keyframes fp6-head-step2 {
    62%, 74% { border-top-color: #3b82f6; }
    0%, 60%, 76%, 100% { border-top-color: #cbd5e1; }
  }


  /* ── STEP 3 (Output) ── */
  .fp6-step3-box { animation: fp6-loop-step3-box 11s infinite ease-in-out; }
  @keyframes fp6-loop-step3-box {
    72%, 94% {
      border-color: #3b82f6; color: #1e40af;
      background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
      box-shadow: 0 8px 24px rgba(59, 130, 246, 0.25);
    }
    0%, 70%, 96%, 100% {
      border-color: #e2e8f0; color: #64748b; background: #ffffff;
      box-shadow: 0 4px 10px rgba(0,0,0,0.03);
    }
  }

  /* Step 3 Bullets */
  .fp6-step3-b1 { animation: fp6-bullet-step3-b1 11s infinite ease-in-out; }
  .fp6-step3-b2 { animation: fp6-bullet-step3-b2 11s infinite ease-in-out; }

  @keyframes fp6-bullet-step3-b1 {
    72%, 94% { font-weight: 900; color: inherit; }
    0%, 70%, 96%, 100% { color: #94a3b8; font-weight: 600; }
  }
  @keyframes fp6-bullet-step3-b2 {
    80%, 94% { font-weight: 900; color: inherit; }
    0%, 78%, 96%, 100% { color: #94a3b8; font-weight: 600; }
  }

  /* Step 3 Dots */
  .fp6-step3-d1 { animation: fp6-dot-step3-d1 11s infinite ease-in-out; }
  .fp6-step3-d2 { animation: fp6-dot-step3-d2 11s infinite ease-in-out; }

  @keyframes fp6-dot-step3-d1 {
    72%, 94% { background-color: currentColor; transform: scale(1.2); }
    0%, 70%, 96%, 100% { background-color: transparent; transform: scale(1); }
  }
  @keyframes fp6-dot-step3-d2 {
    80%, 94% { background-color: currentColor; transform: scale(1.2); }
    0%, 78%, 96%, 100% { background-color: transparent; transform: scale(1); }
  }


  /* ── keyframes ───────────────────────────────────────────────────────── */
  @keyframes fp6-fadein {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes fp6-flow-v {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(250%); }
  }
  @keyframes fp6-sheen {
    0% { left: -75%; }
    100% { left: 125%; }
  }

  /* ── root connector (straight down) ─────────────────────────────────── */
  .fp6-root-vc {
    display: flex; flex-direction: column; align-items: center;
    padding: 6px 0;
  }
  .fp6-root-vc-line {
    width: 3.5px; height: 30px;
    position: relative; overflow: hidden;
    background: #cbd5e1;
  }
  .fp6-root-vc-shine {
    position: absolute; top: -100%; left: 0;
    width: 100%; height: 50%;
    background: linear-gradient(to bottom, transparent, #0ea5e9, transparent);
    animation: fp6-flow-v 2s linear infinite;
  }
  .fp6-root-vc-head {
    width: 0; height: 0;
    border-left: 7px solid transparent;
    border-right: 7px solid transparent;
    border-top: 10px solid #0ea5e9;
    margin-top: -1px;
  }
</style>

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<div class="fp6">
  <div class="fp6-title">AI Pipeline — PCB Defect Detection System</div>

  <div class="fp6-pipeline">

    <!-- FEEDBACK LOOP PATH (C-shape: Output right → up → Data Acquisition right) -->
    <div class="fp6-loop-path">
      <div class="fp6-loop-arrowhead"></div>
      <div class="fp6-loop-label">LOOP</div>
    </div>

    <!-- ROOT NODE: Data Acquisition -->
    <div class="fp6-root-box">Data Acquisition</div>

    <!-- connector down -->
    <div class="fp6-root-vc">
      <div class="fp6-root-vc-line"><div class="fp6-root-vc-shine"></div></div>
      <div class="fp6-root-vc-head"></div>
    </div>

    <!-- STEP 1: Pre-processing -->
    <div class="fp6-ng">
      <div class="fp6-node-row">
        <div class="fp6-nbox fp6-step1-box">Pre-<br>processing</div>
        <div class="fp6-blist">
          <div class="fp6-b fp6-step1-b1"><span class="fp6-b-dot fp6-step1-d1"></span>Letterboxing &amp; Resizing</div>
          <div class="fp6-b fp6-step1-b2"><span class="fp6-b-dot fp6-step1-d2"></span>Color Space Conversion</div>
          <div class="fp6-b fp6-step1-b3"><span class="fp6-b-dot fp6-step1-d3"></span>Normalization &amp; Permutation</div>
        </div>
      </div>
    </div>
    <div class="fp6-vc"><div class="fp6-vc-line fp6-step1-vc-line"><div class="fp6-vc-shine fp6-step1-vc-shine"></div></div><div class="fp6-vc-head fp6-step1-vc-head"></div></div>

    <!-- STEP 2: Model Inference -->
    <div class="fp6-ng">
      <div class="fp6-node-row">
        <div class="fp6-nbox fp6-step2-box">Model<br>Inference</div>
        <div class="fp6-blist">
          <div class="fp6-b fp6-step2-b1"><span class="fp6-b-dot fp6-step2-d1"></span>Tensor Injection</div>
          <div class="fp6-b fp6-step2-b2"><span class="fp6-b-dot fp6-step2-d2"></span>Forward Propagation</div>
          <div class="fp6-b fp6-step2-b3"><span class="fp6-b-dot fp6-step2-d3"></span>Predicts Generation</div>
        </div>
      </div>
    </div>
    <div class="fp6-vc"><div class="fp6-vc-line fp6-step2-vc-line"><div class="fp6-vc-shine fp6-step2-vc-shine"></div></div><div class="fp6-vc-head fp6-step2-vc-head"></div></div>

    <!-- STEP 3: Output -->
    <div class="fp6-ng">
      <div class="fp6-node-row">
        <div class="fp6-nbox fp6-step3-box">Output</div>
        <div class="fp6-blist">
          <div class="fp6-b fp6-step3-b1"><span class="fp6-b-dot fp6-step3-d1"></span>UI Rendering</div>
          <div class="fp6-b fp6-step3-b2"><span class="fp6-b-dot fp6-step3-d2"></span>Database Logging</div>
        </div>
      </div>
    </div>

  </div><!-- end fp6-pipeline -->
</div><!-- end fp6 -->
"""


def render():
    with gr.Column():
        gr.HTML(value=_HTML)

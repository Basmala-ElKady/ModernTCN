"""Generate presentation slides in PDF format using ReportLab (Landscape 16:9).

Designed for the 5-minute presentation and oral exam in Lab 5.
"""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
)

def build_slides(output_path: str | Path = "slides/slides.pdf"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Landscape letter format (11 x 8.5 inches)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(letter),
        leftMargin=40,
        rightMargin=40,
        topMargin=35,
        bottomMargin=35,
    )

    styles = getSampleStyleSheet()

    slide_title_style = ParagraphStyle(
        'SlideTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0d47a1'),
        spaceAfter=12,
    )

    main_title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        textColor=colors.HexColor('#1a237e'),
        alignment=1, # Center
        spaceAfter=15,
    )

    main_subtitle_style = ParagraphStyle(
        'MainSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=13,
        leading=18,
        textColor=colors.HexColor('#37474f'),
        alignment=1, # Center
        spaceAfter=25,
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=16,
        textColor=colors.HexColor('#212121'),
        spaceAfter=8,
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=16,
        textColor=colors.HexColor('#1b5e20'),
        spaceAfter=10,
    )

    table_text_style = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.whitesmoke,
    )

    story = []

    # ================= SLIDE 1 =================
    story.append(Spacer(1, 40))
    story.append(Paragraph("ModernTCN: Large Kernels Instead of Dilation", main_title_style))
    story.append(Paragraph("A Modern Pure Convolution Structure for General Time Series Analysis<br/><b>Topic 6 | Deep Learning (AIE332) | Alamein International University</b>", main_subtitle_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Presentation Outline (5-min oral defense)</b>:<br/>"
                           "• Motivation: The Dilation Paradox & Effective Receptive Field Decay<br/>"
                           "• Architecture: Modern Pure Convolutions (Patching, Large DWConv, Decoupled FFNs, RevIN)<br/>"
                           "• Quantitative Claim Validation & Multi-Seed Benchmark Results<br/>"
                           "• Depthwise Kernel Size Ablation & Autograd ERF Analysis<br/>"
                           "• Reproducible Failure Case: TCN Collapse on Long-Horizon ETTm1", bullet_style))
    story.append(PageBreak())

    # ================= SLIDE 2 =================
    story.append(Paragraph("Motivation: Why Replace Dilated Convolutions?", slide_title_style))
    story.append(Paragraph("<b>The Lab 4 TCN Baseline (Bai et al., 2018)</b>:", bullet_style))
    story.append(Paragraph("• Relies on small kernels (k=3) and exponential dilation (d = 1, 2, 4, 8, 16, ...).", bullet_style))
    story.append(Paragraph("<b>The Dilation Paradox (Gridding / Receptive Field Holes)</b>:", bullet_style))
    story.append(Paragraph("• Theoretical receptive field grows geometrically, but intermediate timesteps are skipped!<br/>"
                           "• High-frequency fluctuations slip between dilation points, creating blind spots.", bullet_style))
    story.append(Paragraph("<b>Effective Receptive Field (ERF) Decay</b>:", bullet_style))
    story.append(Paragraph("• Autograd sensitivity degrades exponentially away from the prediction point.", bullet_style))
    story.append(Paragraph("<b>Missing Cross-Variable Interaction</b>:", bullet_style))
    story.append(Paragraph("• Standard TCN processes channels independently; cannot learn physical coupling between series (e.g. electrical load vs oil temperature).", bullet_style))
    story.append(PageBreak())

    # ================= SLIDE 3 =================
    story.append(Paragraph("The ModernTCN Architecture: Modernizing Pure Convolutions", slide_title_style))
    story.append(Paragraph("1. <b>Reversible Instance Normalization (RevIN)</b>: Normalizes window mean and variance to handle non-stationarity; denormalizes predictions.", bullet_style))
    story.append(Paragraph("2. <b>Patch Embedding (P=16, S=8)</b>: Aggregates continuous raw timesteps into patch tokens. Suppresses noise, shortens sequence (336 -> 41 tokens), and boosts compute speed.", bullet_style))
    story.append(Paragraph("3. <b>Large Depthwise Conv (K=31) + Small Branch (k=5)</b>: Large kernel captures long-range context without dilation holes; small branch captures local sharp changes. Depthwise design scales parameter count linearly O(D·K).", bullet_style))
    story.append(Paragraph("4. <b>Decoupled Pointwise (1x1) FFNs</b>:<br/>"
                           "   • <i>Intra-variable FFN</i>: Mixes representation channels within each variable.<br/>"
                           "   • <i>Inter-variable FFN</i>: Mixes representations across variables to capture multivariate physical relationships.", bullet_style))
    story.append(PageBreak())

    # ================= SLIDE 4 =================
    story.append(Paragraph("Experimental Setup & Benchmark Protocol", slide_title_style))
    story.append(Paragraph("• <b>Datasets</b>: ETTh1 (hourly electricity transformer temperature) & ETTm1 (15-minute resolution).", bullet_style))
    story.append(Paragraph("• <b>Splits</b>: Standard chronological Informer splits (12/4/4 months for train/val/test). Scaler fit strictly on train split.", bullet_style))
    story.append(Paragraph("• <b>Input Window & Horizons</b>: L = 336 steps; Forecast horizons H in {96, 336} steps.", bullet_style))
    story.append(Paragraph("• <b>Parameter Parity</b>: ModernTCN (~65K params) matched against DLinear (~65K params) and Lab 4 TCN (~207K params).", bullet_style))
    story.append(Paragraph("• <b>Multi-Seed Rigor</b>: 3 independent seeds (42, 123, 2026); metrics reported as Mean ± Sample Standard Deviation (ddof=1).", bullet_style))
    story.append(Paragraph("• <b>Model Selection</b>: Checkpoints selected on validation MSE; test set evaluated exactly once.", bullet_style))
    story.append(PageBreak())

    # ================= SLIDE 5 =================
    story.append(Paragraph("Quantitative Claim Verification & Benchmark Results", slide_title_style))
    story.append(Paragraph("<b>Paper Claim (Luo & Wang, ICLR 2024)</b>: ModernTCN outperforms conventional TCNs and competitive linear models (DLinear) across ETT benchmarks.", callout_style))

    bdata = [
        [
            Paragraph("<b>Dataset</b>", table_header_style), Paragraph("<b>Horizon</b>", table_header_style),
            Paragraph("<b>ModernTCN Test MSE</b>", table_header_style), Paragraph("<b>DLinear Test MSE</b>", table_header_style),
            Paragraph("<b>TCN Baseline Test MSE</b>", table_header_style), Paragraph("<b>ModernTCN vs TCN</b>", table_header_style),
        ],
        [
            Paragraph("ETTh1", table_text_style), Paragraph("96", table_text_style),
            Paragraph("<b>0.3799 ± 0.0023</b>", table_text_style), Paragraph("0.3852 ± 0.0045", table_text_style),
            Paragraph("0.7789 ± 0.0395", table_text_style), Paragraph("<b>-51.2% MSE</b>", table_text_style),
        ],
        [
            Paragraph("ETTh1", table_text_style), Paragraph("336", table_text_style),
            Paragraph("<b>0.4481 ± 0.0069</b>", table_text_style), Paragraph("0.4691 ± 0.0165", table_text_style),
            Paragraph("0.9806 ± 0.0393", table_text_style), Paragraph("<b>-54.3% MSE</b>", table_text_style),
        ],
        [
            Paragraph("ETTm1", table_text_style), Paragraph("96", table_text_style),
            Paragraph("<b>0.2970 ± 0.0044</b>", table_text_style), Paragraph("0.3078 ± 0.0013", table_text_style),
            Paragraph("0.5375 ± 0.0202", table_text_style), Paragraph("<b>-44.7% MSE</b>", table_text_style),
        ],
        [
            Paragraph("ETTm1", table_text_style), Paragraph("336", table_text_style),
            Paragraph("<b>0.3745 ± 0.0062</b>", table_text_style), Paragraph("0.3872 ± 0.0063", table_text_style),
            Paragraph("0.9611 ± 0.0195", table_text_style), Paragraph("<b>-61.0% MSE</b>", table_text_style),
        ],
    ]
    bt = Table(bdata, colWidths=[80, 70, 150, 140, 150, 110])
    bt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1565c0')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#90caf9')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(bt)
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Verdict: STRONGLY SUPPORTED.</b> ModernTCN consistently achieves the lowest MSE and MAE on all 4 benchmark tasks.", callout_style))
    story.append(PageBreak())

    # ================= SLIDE 6 =================
    story.append(Paragraph("Depthwise Kernel Size Ablation Study", slide_title_style))
    story.append(Paragraph("<b>Topic 6 Requirement (e)</b>: Evaluate 3 kernel sizes at matched parameter count (<6% variation).", bullet_style))

    kdata = [
        [
            Paragraph("<b>Kernel Size (K)</b>", table_header_style), Paragraph("<b>Total Parameters</b>", table_header_style),
            Paragraph("<b>Test MSE (mean ± std)</b>", table_header_style), Paragraph("<b>Test MAE (mean ± std)</b>", table_header_style),
            Paragraph("<b>Best Val MSE</b>", table_header_style),
        ],
        [
            Paragraph("<b>15</b>", table_text_style), Paragraph("62,190 (-4.7%)", table_text_style),
            Paragraph("<b>0.3788 ± 0.0013</b>", table_text_style), Paragraph("<b>0.4024 ± 0.0014</b>", table_text_style),
            Paragraph("0.6853", table_text_style),
        ],
        [
            Paragraph("<b>31</b> (Baseline)", table_text_style), Paragraph("65,262 (0.0%)", table_text_style),
            Paragraph("0.3799 ± 0.0023", table_text_style), Paragraph("0.4032 ± 0.0014", table_text_style),
            Paragraph("0.6882", table_text_style),
        ],
        [
            Paragraph("<b>51</b>", table_text_style), Paragraph("69,102 (+5.9%)", table_text_style),
            Paragraph("0.3830 ± 0.0039", table_text_style), Paragraph("0.4053 ± 0.0027", table_text_style),
            Paragraph("0.6916", table_text_style),
        ],
    ]
    kt = Table(kdata, colWidths=[120, 130, 160, 150, 120])
    kt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00695c')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#80cbc4')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e0f2f1')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(kt)
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Key Takeaways</b>:<br/>"
                           "• Accuracy does not continue to increase past K=15/31: K=15 slightly outperforms K=31 and K=51.<br/>"
                           "• In patched representations (patch stride 8), K=15 already covers the primary temporal cycle of ETTh1. "
                           "Larger kernels add extra parameters without providing useful incremental receptive field.", bullet_style))
    story.append(PageBreak())

    # ================= SLIDE 7 =================
    story.append(Paragraph("Effective Receptive Field (ERF) via Autograd Sensitivity", slide_title_style))
    story.append(Paragraph("• Measured by backpropagating output prediction gradients to input timesteps across multiple horizon points.", bullet_style))
    story.append(Paragraph("• <b>Gradient Mass Concentration</b>: For ModernTCN (K=15), 50% of gradient mass is within 14 timesteps; 90% within 61 timesteps.", bullet_style))
    
    img_path = Path("results/erf_analysis/moderntcn_etth1_h96_kernel15_erf.png")
    if img_path.exists():
        story.append(Image(str(img_path), width=480, height=180))
    story.append(PageBreak())

    # ================= SLIDE 8 =================
    story.append(Paragraph("Reproducible Failure Case: TCN on ETTm1 (H=336)", slide_title_style))
    story.append(Paragraph("<b>The Failure</b>: On 15-minute resolution ETTm1 with horizon 336 (3.5 days), TCN collapses: Test MSE = <b>0.9611 ± 0.0195</b> vs ModernTCN = <b>0.3745 ± 0.0062</b> (2.56x higher error!).", callout_style))
    story.append(Paragraph("<b>Per-Window Error Breakdown (11,185 test windows)</b>:<br/>"
                           "• TCN produces higher MSE on <b>over 98.8% of test windows</b> (11,086 / 11,185).<br/>"
                           "• In worst windows during sharp load transitions, TCN error is <b>6.2x to 9.5x worse</b> than ModernTCN.", bullet_style))
    story.append(Paragraph("<b>Physical & Architectural Explanation</b>:<br/>"
                           "1. <i>Dilation Gridding</i>: 15-min series has rapid cycles. Deep dilation factors skip crucial transition points.<br/>"
                           "2. <i>Univariate Blindness</i>: Oil temperature (OT) physically depends on current load (HUFL, HULL). TCN models variables separately; ModernTCN's inter-variable FFN mixes channels, preserving physical consistency.", bullet_style))
    story.append(PageBreak())

    # ================= SLIDE 9 =================
    story.append(Paragraph("Oral Defense Preparation: Anticipated Lab 5 Questions", slide_title_style))
    story.append(Paragraph("<b>Q1: Explain any line of your code</b><br/>"
                           "• <i>Patching</i>: `Conv1d(1, embed_dim, kernel_size=16, stride=8)` reshaped across [B*C, 1, L].<br/>"
                           "• <i>Depthwise Conv</i>: `Conv1d(embed_dim, embed_dim, kernel_size=31, groups=embed_dim, padding=causal_pad)`.", bullet_style))
    story.append(Paragraph("<b>Q2: Predict the effect of a change before running live</b><br/>"
                           "• <i>Removing RevIN</i>: Test MSE will increase by ~15-20% due to window-level distribution shifts.<br/>"
                           "• <i>Disabling Inter-Variable FFN</i>: Multivariate accuracy will drop substantially on long horizons due to loss of physical cross-channel relationships.", bullet_style))
    story.append(Paragraph("<b>Q3: What result would make you change your conclusion?</b><br/>"
                           "• If a dilated TCN with matched parameter count and continuous dense sampling surpassed large-kernel convolutions on fine-grained long horizons.", bullet_style))
    story.append(PageBreak())

    # ================= SLIDE 10 =================
    story.append(Paragraph("Conclusions & Academic Integrity Record", slide_title_style))
    story.append(Paragraph("<b>Summary of Findings</b>:<br/>"
                           "1. Large depthwise convolutions (K=15, 31) combined with patch tokenization consistently outperform dilated TCNs and linear models.<br/>"
                           "2. Accuracy plateaus once the effective receptive field matches the primary seasonal period.<br/>"
                           "3. Decoupled pointwise FFNs prevent catastrophic multivariate failure under high-frequency conditions.", bullet_style))
    story.append(Paragraph("<b>Compliance & Integrity</b>:<br/>"
                           "• Core models, training, evaluation, and ERF autograd implemented independently from scratch in PyTorch.<br/>"
                           "• Inspected author code: `luodhhh/ModernTCN` and `cure-lab/LTSF-Linear` (recorded in report).<br/>"
                           "• AI assistance declared for debugging, plotting, and ReportLab formatting.", bullet_style))

    doc.build(story)
    print(f"Slides generated successfully: {output_path}")

if __name__ == '__main__':
    build_slides()

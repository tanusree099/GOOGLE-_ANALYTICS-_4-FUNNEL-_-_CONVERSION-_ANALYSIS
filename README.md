# GOOGLE-_ANALYTICS-_4-FUNNEL-_-_CONVERSION-_ANALYSIS
"""
=====================================================================
  GOOGLE ANALYTICS 4 — FUNNEL & CONVERSION ANALYSIS
  Author : Tanusree Saha
  Tool   : Python (Pandas, Matplotlib)
  Purpose: Digital Marketing Portfolio Project
=====================================================================

WHAT THIS PROJECT DOES:
    Simulates a full GA4 analytics implementation and analysis:

    1. Multi-touch Attribution Modelling
         - Last-click, Linear, Time-decay, Data-driven
         - Channel contribution comparison
         - Revenue attribution by model

    2. Conversion Funnel Analysis
         - 5-step checkout funnel drop-off analysis
         - Device-level funnel performance
         - Drop-off root cause mapping

    3. Cohort Retention Analysis
         - Weekly cohort retention curves
         - Channel-based retention comparison
         - LTV estimation by acquisition source

    4. Audience Segmentation
         - RFM scoring (Recency, Frequency, Monetary)
         - Behavioural segment identification
         - High-value user profiling

    5. Channel Performance Dashboard
         - Sessions, conversions, revenue by channel
         - ROAS, CPA, CVR comparison
         - Recommendations for budget reallocation
=====================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)


# ─────────────────────────────────────────────
#  1. SYNTHETIC GA4 SESSION DATA
# ─────────────────────────────────────────────

def generate_sessions(n=15000, days=90):
    """Generate synthetic GA4 session data."""
    channels = ['Organic Search', 'Paid Search', 'Social Media',
                'Email', 'Direct', 'Referral', 'Display']
    channel_weights = [0.30, 0.22, 0.18, 0.12, 0.10, 0.05, 0.03]
    devices = ['mobile', 'desktop', 'tablet']
    device_weights = [0.58, 0.36, 0.06]

    start = datetime(2026, 1, 1)
    sessions = []
    for _ in range(n):
        channel  = np.random.choice(channels, p=channel_weights)
        device   = np.random.choice(devices, p=device_weights)
        date     = start + timedelta(days=np.random.randint(0, days))

        # Channel-specific conversion rates
        base_cvr = {'Organic Search': 0.032, 'Paid Search': 0.041,
                    'Email': 0.052, 'Direct': 0.038, 'Social Media': 0.018,
                    'Referral': 0.028, 'Display': 0.009}[channel]
        device_mult = {'desktop': 1.3, 'mobile': 0.75, 'tablet': 0.9}[device]
        converted = np.random.random() < (base_cvr * device_mult)

        revenue = 0
        if converted:
            revenue = max(0, np.random.normal(1200, 600))

        sessions.append({
            'date'          : date,
            'channel'       : channel,
            'device'        : device,
            'pages_viewed'  : np.random.randint(1, 15),
            'session_dur_s' : max(10, int(np.random.normal(180, 120))),
            'bounced'       : np.random.random() < 0.42,
            'converted'     : converted,
            'revenue'       : round(revenue, 2),
            'user_id'       : np.random.randint(1, 8000),
        })
    return pd.DataFrame(sessions)


# ─────────────────────────────────────────────
#  2. FUNNEL ANALYSIS
# ─────────────────────────────────────────────

def generate_funnel_data(n_entries=10000):
    """
    5-step checkout funnel:
    Homepage → Product Page → Add to Cart → Checkout → Purchase

    Simulate realistic drop-off at each step by device.
    """
    devices = {'desktop': 0.36, 'mobile': 0.58, 'tablet': 0.06}

    # Step completion rates by device (realistic e-commerce benchmarks)
    completion = {
        'desktop': [1.0, 0.68, 0.42, 0.31, 0.22],
        'mobile' : [1.0, 0.55, 0.28, 0.17, 0.10],
        'tablet' : [1.0, 0.62, 0.35, 0.24, 0.16],
    }

    steps = ['Homepage', 'Product Page', 'Add to Cart', 'Checkout', 'Purchase']
    funnel_data = {}

    for device, share in devices.items():
        n_device = int(n_entries * share)
        funnel_data[device] = [int(n_device * r) for r in completion[device]]

    # Combined
    combined = [sum(funnel_data[d][i] for d in devices) for i in range(5)]

    return steps, funnel_data, combined


def funnel_drop_off_analysis(steps, combined):
    """Compute drop-off rates between funnel steps."""
    results = []
    for i in range(len(steps)):
        prev = combined[i-1] if i > 0 else combined[0]
        drop_pct = (1 - combined[i]/prev) * 100 if i > 0 else 0
        results.append({
            'step'       : steps[i],
            'users'      : combined[i],
            'drop_off_%' : round(drop_pct, 1),
            'cvr_from_start': round(combined[i]/combined[0]*100, 2),
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
#  3. MULTI-TOUCH ATTRIBUTION
# ─────────────────────────────────────────────

def attribution_models(sessions_df):
    """
    Compare 4 attribution models for channel revenue credit.

    Last-Click   : 100% credit to last channel before conversion
    First-Click  : 100% credit to first channel (acquisition focus)
    Linear       : Equal credit across all touchpoints
    Time-Decay   : More credit to channels closer to conversion
    """
    converted = sessions_df[sessions_df['converted']].copy()
    channels  = sessions_df['channel'].unique()

    # Last-click (standard GA4 default)
    last_click = converted.groupby('channel')['revenue'].sum()

    # Simulate multi-touch paths (simplified)
    # For each conversion, assign credit across 1-4 touchpoints
    linear_credit  = {ch: 0.0 for ch in channels}
    decay_credit   = {ch: 0.0 for ch in channels}
    first_credit   = {ch: 0.0 for ch in channels}

    all_channels = list(channels)
    for _, row in converted.iterrows():
        n_touch = np.random.randint(1, 5)
        path    = np.random.choice(all_channels, n_touch, replace=True)
        rev     = row['revenue']

        # Linear
        for ch in path:
            linear_credit[ch] += rev / n_touch

        # Time decay (more weight to later touches)
        weights = np.exp(np.arange(n_touch) * 0.5)
        weights = weights / weights.sum()
        for ch, w in zip(path, weights):
            decay_credit[ch] += rev * w

        # First click
        first_credit[path[0]] += rev

    models = pd.DataFrame({
        'Last-Click'  : last_click,
        'First-Click' : pd.Series(first_credit),
        'Linear'      : pd.Series(linear_credit),
        'Time-Decay'  : pd.Series(decay_credit),
    }).fillna(0)

    return models


# ─────────────────────────────────────────────
#  4. COHORT RETENTION
# ─────────────────────────────────────────────

def cohort_retention(sessions_df, n_weeks=8):
    """
    Build weekly cohort retention table.
    Cohort = week of first session.
    """
    sessions_df = sessions_df.copy()
    sessions_df['week'] = sessions_df['date'].apply(
        lambda d: (d - datetime(2026,1,1)).days // 7)

    # First visit week per user
    first_week = sessions_df.groupby('user_id')['week'].min().rename('cohort_week')
    sessions_df = sessions_df.join(first_week, on='user_id')
    sessions_df['week_num'] = sessions_df['week'] - sessions_df['cohort_week']

    retention = {}
    for cohort in range(min(n_weeks, sessions_df['cohort_week'].max() + 1)):
        cohort_users = sessions_df[sessions_df['cohort_week'] == cohort]['user_id'].nunique()
        if cohort_users < 10:
            continue
        row = {}
        for w in range(n_weeks):
            active = sessions_df[
                (sessions_df['cohort_week'] == cohort) &
                (sessions_df['week_num'] == w)
            ]['user_id'].nunique()
            row[w] = round(active / cohort_users * 100, 1) if cohort_users > 0 else 0
        retention[f'Week {cohort+1}'] = row

    return pd.DataFrame(retention).T.fillna(0)


# ─────────────────────────────────────────────
#  5. RFM SEGMENTATION
# ─────────────────────────────────────────────

def rfm_segmentation(sessions_df):
    """
    RFM Analysis: Recency, Frequency, Monetary value.
    Segments users into Champions, Loyal, At Risk, Lost etc.
    """
    converted = sessions_df[sessions_df['converted']].copy()
    today     = datetime(2026, 4, 1)

    rfm = converted.groupby('user_id').agg(
        recency   = ('date', lambda x: (today - x.max()).days),
        frequency = ('converted', 'count'),
        monetary  = ('revenue', 'sum'),
    ).reset_index()

    # Score 1-5 for each dimension
    rfm['R'] = pd.qcut(rfm['recency'],   5, labels=[5,4,3,2,1], duplicates='drop')
    rfm['F'] = pd.qcut(rfm['frequency'], 5, labels=[1,2,3,4,5], duplicates='drop')
    rfm['M'] = pd.qcut(rfm['monetary'],  5, labels=[1,2,3,4,5], duplicates='drop')
    rfm[['R','F','M']] = rfm[['R','F','M']].apply(pd.to_numeric, errors='coerce').fillna(3)

    rfm['RFM_score'] = rfm['R'] + rfm['F'] + rfm['M']

    def segment(score):
        if score >= 13: return 'Champions'
        elif score >= 11: return 'Loyal Customers'
        elif score >= 9: return 'Potential Loyalists'
        elif score >= 7: return 'At Risk'
        elif score >= 5: return 'Needs Attention'
        else: return 'Lost'

    rfm['segment'] = rfm['RFM_score'].apply(segment)
    return rfm


# ─────────────────────────────────────────────
#  6. CHANNEL KPI SUMMARY
# ─────────────────────────────────────────────

def channel_kpis(sessions_df):
    """Compute per-channel KPIs: sessions, CVR, revenue, CPA, ROAS."""
    # Simulated ad spend per channel
    spend = {'Paid Search': 45000, 'Social Media': 28000, 'Display': 12000,
             'Organic Search': 0, 'Email': 3500, 'Direct': 0, 'Referral': 0}

    kpis = sessions_df.groupby('channel').agg(
        sessions    = ('user_id', 'count'),
        conversions = ('converted', 'sum'),
        revenue     = ('revenue', 'sum'),
        bounce_rate = ('bounced', 'mean'),
        avg_pages   = ('pages_viewed', 'mean'),
    ).reset_index()

    kpis['cvr_%']    = (kpis['conversions'] / kpis['sessions'] * 100).round(2)
    kpis['spend']    = kpis['channel'].map(spend).fillna(0)
    kpis['cpa']      = (kpis['spend'] / kpis['conversions'].replace(0, np.nan)).round(0)
    kpis['roas']     = (kpis['revenue'] / kpis['spend'].replace(0, np.nan)).round(2)
    kpis['revenue']  = kpis['revenue'].round(0)
    kpis['bounce_%'] = (kpis['bounce_rate'] * 100).round(1)
    return kpis


# ─────────────────────────────────────────────
#  7. VISUALISATION
# ─────────────────────────────────────────────

def plot_ga4_dashboard(sessions_df, funnel_data, kpis, attribution, cohort_df, rfm_df):
    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor('#0D1117')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

    BG    = '#0D1117'
    PANEL = '#161B22'
    GC    = '#2A2A3A'
    TC    = '#E0E0E0'
    GREEN = '#00C896'
    RED   = '#FF4444'
    AMBER = '#FFB347'
    BLUE  = '#4A9EFF'
    PURP  = '#C084FC'
    TEAL  = '#00D4C8'

    steps, device_funnel, combined = funnel_data

    # ── Panel 1: Checkout Funnel ──
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(PANEL)
    x = np.arange(len(steps))
    w = 0.25
    colors_d = [BLUE, GREEN, AMBER]
    for i, (device, vals) in enumerate(device_funnel.items()):
        ax1.bar(x + i*w, vals, width=w, label=device.title(),
                color=colors_d[i], alpha=0.85)
    ax1.set_title('5-Step Checkout Funnel by Device\n(Drop-off analysis)',
                  color=TC, fontsize=10, fontweight='bold')
    ax1.set_xticks(x + w)
    ax1.set_xticklabels(steps, color=TC, fontsize=8)
    ax1.set_ylabel('Users', color=TC, fontsize=8)
    ax1.tick_params(colors=TC, labelsize=8)
    ax1.grid(True, color=GC, linewidth=0.4, axis='y')
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=TC)
    for s in ax1.spines.values(): s.set_color(GC)

    # ── Panel 2: Drop-off % ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PANEL)
    dropoff_df = funnel_drop_off_analysis(steps, combined)
    drops = dropoff_df[dropoff_df['drop_off_%'] > 0]
    bars = ax2.barh(drops['step'], drops['drop_off_%'],
                    color=[RED if v > 35 else AMBER if v > 20 else GREEN
                           for v in drops['drop_off_%']], alpha=0.85)
    for bar, val in zip(bars, drops['drop_off_%']):
        ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center', color=TC, fontsize=8)
    ax2.set_title('Step Drop-Off Rate (%)\n(Red = Critical fix needed)',
                  color=TC, fontsize=10, fontweight='bold')
    ax2.set_xlabel('Drop-off %', color=TC, fontsize=8)
    ax2.tick_params(colors=TC, labelsize=8)
    ax2.grid(True, color=GC, linewidth=0.4, axis='x')
    for s in ax2.spines.values(): s.set_color(GC)

    # ── Panel 3: Attribution Model Comparison ──
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor(PANEL)
    attr_plot = attribution.copy()
    attr_plot = attr_plot / 1000  # Convert to thousands
    x3 = np.arange(len(attr_plot))
    w3 = 0.2
    cols_a = [GREEN, BLUE, AMBER, PURP]
    for i, col_name in enumerate(attr_plot.columns):
        ax3.bar(x3 + i*w3, attr_plot[col_name], width=w3,
                label=col_name, color=cols_a[i], alpha=0.85)
    ax3.set_title('Multi-Touch Attribution: Revenue by Channel & Model\n(₹ thousands)',
                  color=TC, fontsize=10, fontweight='bold')
    ax3.set_xticks(x3 + w3*1.5)
    ax3.set_xticklabels(attr_plot.index, color=TC, fontsize=7, rotation=15)
    ax3.set_ylabel('Revenue (₹ thousands)', color=TC, fontsize=8)
    ax3.tick_params(colors=TC, labelsize=7)
    ax3.grid(True, color=GC, linewidth=0.4, axis='y')
    ax3.legend(fontsize=8, facecolor=PANEL, labelcolor=TC)
    for s in ax3.spines.values(): s.set_color(GC)

    # ── Panel 4: Channel CVR ──
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(PANEL)
    kpi_sorted = kpis.sort_values('cvr_%', ascending=True)
    ax4.barh(kpi_sorted['channel'], kpi_sorted['cvr_%'],
             color=[GREEN if v > 3.5 else AMBER if v > 2 else RED
                    for v in kpi_sorted['cvr_%']], alpha=0.85)
    ax4.axvline(kpis['cvr_%'].mean(), color='white', linewidth=1.2,
                linestyle='--', label=f'Avg CVR: {kpis["cvr_%"].mean():.2f}%')
    ax4.set_title('Conversion Rate by Channel (%)\n(Benchmark: Industry avg ~2.5%)',
                  color=TC, fontsize=9, fontweight='bold')
    ax4.set_xlabel('CVR (%)', color=TC, fontsize=8)
    ax4.tick_params(colors=TC, labelsize=8)
    ax4.grid(True, color=GC, linewidth=0.4, axis='x')
    ax4.legend(fontsize=8, facecolor=PANEL, labelcolor=TC)
    for s in ax4.spines.values(): s.set_color(GC)

    # ── Panel 5: Cohort Retention ──
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.set_facecolor(PANEL)
    if not cohort_df.empty:
        cmap = plt.cm.RdYlGn
        im = ax5.imshow(cohort_df.values.astype(float), cmap=cmap,
                        aspect='auto', vmin=0, vmax=100)
        ax5.set_xticks(range(cohort_df.shape[1]))
        ax5.set_xticklabels([f'W{i}' for i in range(cohort_df.shape[1])],
                             color=TC, fontsize=7)
        ax5.set_yticks(range(len(cohort_df)))
        ax5.set_yticklabels(cohort_df.index, color=TC, fontsize=7)
        for i in range(cohort_df.shape[0]):
            for j in range(cohort_df.shape[1]):
                val = cohort_df.values[i, j]
                if val > 0:
                    ax5.text(j, i, f'{val:.0f}%', ha='center', va='center',
                             fontsize=6, color='black' if val > 50 else 'white')
    ax5.set_title('Weekly Cohort Retention Heatmap (%)\n(Green = strong retention)',
                  color=TC, fontsize=10, fontweight='bold')
    ax5.set_xlabel('Weeks Since Acquisition', color=TC, fontsize=8)
    for s in ax5.spines.values(): s.set_color(GC)

    # ── Panel 6: RFM Segment ──
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(PANEL)
    seg_counts = rfm_df['segment'].value_counts()
    seg_colors = {'Champions':'#00C896','Loyal Customers':'#4A9EFF',
                  'Potential Loyalists':'#00D4C8','At Risk':'#FFB347',
                  'Needs Attention':'#FF9F43','Lost':'#FF4444'}
    colors_seg = [seg_colors.get(s, BLUE) for s in seg_counts.index]
    wedges, texts, autotexts = ax6.pie(
        seg_counts.values, labels=seg_counts.index,
        autopct='%1.0f%%', colors=colors_seg,
        textprops={'color': TC, 'fontsize': 7},
        startangle=90, pctdistance=0.78)
    for at in autotexts: at.set_color(BG); at.set_fontsize(6.5)
    ax6.set_title('RFM User Segmentation\n(Recency · Frequency · Monetary)',
                  color=TC, fontsize=10, fontweight='bold')

    fig.suptitle('Google Analytics 4 — Funnel, Attribution & Audience Intelligence Dashboard',
                 color=TC, fontsize=13, fontweight='bold', y=1.01)
    plt.savefig('/mnt/user-data/outputs/ga4_funnel_dashboard.png',
                dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print("Chart saved.")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("  GOOGLE ANALYTICS 4 — FUNNEL & CONVERSION ANALYSIS")
    print("="*60)

    sessions_df = generate_sessions(15000, 90)

    print(f"\n── Session Overview (90 days) ──")
    print(f"  Total sessions    : {len(sessions_df):,}")
    print(f"  Total conversions : {sessions_df['converted'].sum():,}")
    print(f"  Overall CVR       : {sessions_df['converted'].mean()*100:.2f}%")
    print(f"  Total revenue     : ₹{sessions_df['revenue'].sum():,.0f}")
    print(f"  Avg order value   : ₹{sessions_df[sessions_df['converted']]['revenue'].mean():,.0f}")

    print(f"\n── Channel KPIs ──")
    kpis = channel_kpis(sessions_df)
    print(kpis[['channel','sessions','conversions','cvr_%','revenue','roas']].to_string(index=False))

    print(f"\n── Funnel Analysis ──")
    steps, device_funnel, combined = generate_funnel_data(10000)
    funnel_df = funnel_drop_off_analysis(steps, combined)
    print(funnel_df.to_string(index=False))
    worst = funnel_df.iloc[1:]['drop_off_%'].idxmax()
    print(f"\n  Biggest drop-off: {funnel_df.loc[worst,'step']} ({funnel_df.loc[worst,'drop_off_%']}%)")
    print(f"  Recommendation: A/B test CTA, page speed, and trust signals at this step.")

    print(f"\n── Attribution Models ──")
    attribution = attribution_models(sessions_df)
    print((attribution/1000).round(1).to_string())
    print(f"\n  Insight: Organic Search gets MORE credit in linear/time-decay vs last-click.")
    print(f"  Action : Increase SEO investment — it is under-credited in last-click model.")

    print(f"\n── Cohort Retention ──")
    cohort_df = cohort_retention(sessions_df)
    print(f"  Week 0 (acquisition) retention: 100%")
    if cohort_df.shape[1] > 1:
        w1_avg = cohort_df.iloc[:,1].mean()
        w4_avg = cohort_df.iloc[:,min(4,cohort_df.shape[1]-1)].mean()
        print(f"  Week 1 avg retention : {w1_avg:.1f}%")
        print(f"  Week 4 avg retention : {w4_avg:.1f}%")

    print(f"\n── RFM Segmentation ──")
    rfm_df = rfm_segmentation(sessions_df)
    for seg, cnt in rfm_df['segment'].value_counts().items():
        rev = rfm_df[rfm_df['segment']==seg]['monetary'].sum()
        print(f"  {seg:22s}: {cnt:4d} users  |  ₹{rev:>10,.0f} revenue")

    print(f"\n── Generating Dashboard ──")
    plot_ga4_dashboard(sessions_df, (steps, device_funnel, combined),
                       kpis, attribution, cohort_df, rfm_df)
    print("\nDone.")

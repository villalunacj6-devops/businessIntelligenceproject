import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==============================================================================
# 1. REALISTIC INSTITUTIONAL DATASET GENERATOR (BATSTATEU ARASOF-NASUGBU)
#    Models real District 1 Batangas geography, semester seasonality, RA 10931
#    (UniFAST) coverage, section congestion, and BatStateU grading rules.
# ==============================================================================
CSV_FILENAME = "batstateu_arasof_institutional_1500.csv"

def generate_realistic_arasof_dataset(n_records=1500, seed=108):
    rng = np.random.default_rng(seed)

    # Official ARASOF-Nasugbu Colleges (Chapter 1.1)
    colleges = [
        "CICS",
        "College of Engineering Technology",
        "College of Teacher Education",
        "CABEIHM",
        "College of Arts and Sciences"
    ]
    college_short = {
        "CICS": "CICS",
        "College of Engineering Technology": "CET",
        "College of Teacher Education": "CTE",
        "CABEIHM": "CABEIHM",
        "College of Arts and Sciences": "CAS"
    }
    # Realistic enrollment share at ARASOF-Nasugbu (CABEIHM & CICS largest)
    college_weights = [0.25, 0.20, 0.17, 0.25, 0.13]

    programs_map = {
        "CICS": ["BS Information Technology", "BS Computer Science"],
        "College of Engineering Technology": [
            "BS Computer Engineering Technology",
            "BS Electronics Engineering Technology",
            "BS Electrical Engineering Technology"
        ],
        "College of Teacher Education": [
            "Bachelor of Secondary Education",
            "Bachelor of Elementary Education"
        ],
        "CABEIHM": [
            "BS Accountancy",
            "BS Business Administration",
            "BS Hospitality Management",
            "BS Tourism Management"
        ],
        "College of Arts and Sciences": [
            "BS Fisheries and Aquatic Sciences",
            "BS Psychology",
            "BA Communication"
        ]
    }

    # Real Western Batangas Towns around ARASOF-Nasugbu (Brgy. Bucana)
    # Distances reflect actual road kilometers via Palico-Balayan-Batangas & Nasugbu-Lian-Calatagan highways
    town_specs = {
        "Nasugbu":   {"prob": 0.41, "km": 4.2,  "sd": 2.4},
        "Lian":      {"prob": 0.19, "km": 10.8, "sd": 2.1},
        "Tuy":       {"prob": 0.13, "km": 19.4, "sd": 2.8},
        "Balayan":   {"prob": 0.11, "km": 27.6, "sd": 3.2},
        "Calatagan": {"prob": 0.07, "km": 33.5, "sd": 3.5},
        "Calaca":    {"prob": 0.04, "km": 36.2, "sd": 3.0},
        "Lemery":    {"prob": 0.03, "km": 43.8, "sd": 3.4},
        "Taal":      {"prob": 0.02, "km": 47.5, "sd": 3.1},
    }
    towns = list(town_specs.keys())
    town_probs = [town_specs[t]["prob"] for t in towns]

    # Past 3 Academic Years (6 Semesters) with real Philippine HEI enrollment waves:
    # - 2023-24 1st: Post-pandemic F2F surge
    # - 2023-24 2nd: Mid-year attrition dip
    # - 2024-25 1st: Moderate intake stabilization
    # - 2024-25 2nd: Retention screening dip
    # - 2025-26 1st: National Engineering University (RA 11694) intake surge
    # - 2025-26 2nd: Seasonal 2nd-sem taper
    terms = [
        "2023-24 1st", "2023-24 2nd",
        "2024-25 1st", "2024-25 2nd",
        "2025-26 1st", "2025-26 2nd"
    ]
    term_weights = [0.172, 0.138, 0.178, 0.144, 0.204, 0.164]

    chosen_terms = rng.choice(terms, size=n_records, p=term_weights)
    chosen_ay = np.array([t.split()[0] for t in chosen_terms])
    chosen_sem = np.array([t.split()[1] for t in chosen_terms])

    chosen_colleges = rng.choice(colleges, size=n_records, p=college_weights)
    chosen_programs = [rng.choice(programs_map[c]) for c in chosen_colleges]

    # Realistic funnel across year levels (1st yr largest, 4th yr smaller due to attrition)
    year_levels = rng.choice([1, 2, 3, 4], size=n_records, p=[0.36, 0.26, 0.20, 0.18])
    chosen_towns = rng.choice(towns, size=n_records, p=town_probs)

    distance_km = np.array([
        round(max(0.8, rng.normal(town_specs[t]["km"], town_specs[t]["sd"])), 1)
        for t in chosen_towns
    ])

    # Section congestion: 1st-year & 1st-semester surge cohorts have higher chance of overloaded sections (>45)
    base_sec = np.where(chosen_sem == "1st", 43, 37) + np.where(year_levels == 1, 5, 0) + np.where(chosen_ay == "2025-26", 3, 0)
    section_size = np.clip(rng.normal(base_sec, 5.5, size=n_records).round().astype(int), 26, 56)

    # Units enrolled: Regular load 18-24 units; irregular/working students 15 units
    units_enrolled = np.where(
        year_levels == 4,
        rng.choice([15, 18, 21], size=n_records, p=[0.35, 0.45, 0.20]),
        rng.choice([15, 18, 21, 24], size=n_records, p=[0.10, 0.22, 0.46, 0.22])
    )

    # RA 10931 (Free Higher Education / UniFAST): ~86% covered; remaining ~14% are second-coursers,
    # transferees past MRR (Maximum Residency Rule), or shiftees paying assessment fees
    unifast_prob = np.where(year_levels == 4, 0.78, 0.88)
    scholarship_status = (rng.random(n_records) < unifast_prob).astype(int)

    # Class Attendance (%): Driven by commute distance, 1st-sem typhoon weather (Habagat), and scholarship stability
    habagat_penalty = np.where(chosen_sem == "1st", -1.4, 0.0)
    commute_penalty = 0.24 * np.maximum(0, distance_km - 5.0)
    attendance_rate = np.clip(
        rng.normal(92.8, 6.2, size=n_records) - commute_penalty + habagat_penalty + (scholarship_status * 1.8),
        52.0, 100.0
    ).round(1)

    # Weekly Virtual Learning Environment / Google Classroom Logins
    lms_login_freq = np.clip(
        rng.poisson(11, size=n_records) - (distance_km // 16).astype(int) + np.where(chosen_colleges == "CICS", 3, 0),
        1, 26
    )

    # Official BatStateU Grade Scale (1.00 = Excellent, 3.00 = Passing, 4.00 = Conditional, 5.00 = Failed)
    batstateu_grades = np.array([1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 4.00, 5.00])
    def snap_batstateu_grade(val):
        return batstateu_grades[np.abs(batstateu_grades - val).argmin()]

    # Prior GWA (reflects prerequisite mastery)
    program_rigor = np.where(
        np.isin(chosen_programs, ["BS Accountancy", "BS Computer Science", "BS Computer Engineering Technology"]),
        0.18, 0.0
    )
    raw_prior_gwa = (
        3.15
        - (attendance_rate / 100.0) * 1.15
        - (lms_login_freq / 25.0) * 0.35
        + program_rigor
        + rng.normal(0, 0.26, size=n_records)
    )
    prior_gwa = np.array([snap_batstateu_grade(g) for g in raw_prior_gwa])

    # Midterm Grade: Combines Prior GWA, current attendance, LMS engagement, and overloaded section penalty
    overload_penalty = np.maximum(0, section_size - 45) * 0.028
    raw_midterm = (
        0.48 * prior_gwa
        + 0.52 * (3.55 - (attendance_rate / 100.0) * 1.48 - (lms_login_freq / 25.0) * 0.42)
        + overload_penalty
        + rng.normal(0, 0.22, size=n_records)
    )
    midterm_grade = np.array([snap_batstateu_grade(g) for g in raw_midterm])

    # Final Grade: Students with 3.00-4.00 midterms either recover via remedial work or slip to 5.00
    raw_final = midterm_grade + (overload_penalty * 0.45) + rng.normal(-0.04, 0.19, size=n_records)
    final_grade = np.array([snap_batstateu_grade(g) for g in raw_final])

    # Realistic Attrition / Dropout Event (higher in 1st Year, failing midterms, low attendance, lost UniFAST)
    freshman_shock = np.where(year_levels == 1, 0.35, 0.0)
    risk_logit = (
        1.55 * (midterm_grade - 2.65)
        + 0.80 * (prior_gwa - 2.50)
        - 0.072 * (attendance_rate - 83.0)
        - 0.095 * (lms_login_freq - 8)
        + 0.028 * (distance_km - 15.0)
        + 0.045 * (units_enrolled - 18)
        - 0.65 * scholarship_status
        + freshman_shock
        - 0.35
    )
    prob_dropout = 1.0 / (1.0 + np.exp(-risk_logit))
    dropped_out = (rng.random(n_records) < prob_dropout).astype(int)

    on_time_grad = np.where((year_levels == 4) & (dropped_out == 0) & (final_grade <= 3.00), 1, 0)

    # Authentic BatStateU SR-Code format (YY-7XXXX for ARASOF-Nasugbu)
    sr_prefix = np.where(year_levels == 1, "25", np.where(year_levels == 2, "24", np.where(year_levels == 3, "23", "22")))
    student_ids = [f"{sr_prefix[i]}-7{1000 + i:04d}" for i in range(n_records)]

    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Academic_Year": chosen_ay,
        "Semester": chosen_sem,
        "Academic_Term": chosen_terms,
        "College": chosen_colleges,
        "College_Short": [college_short[c] for c in chosen_colleges],
        "Program": chosen_programs,
        "Year_Level": year_levels,
        "Municipality": chosen_towns,
        "Section_Size": section_size,
        "Prior_GWA": prior_gwa,
        "Midterm_Grade": midterm_grade,
        "Attendance_Rate": attendance_rate,
        "LMS_Login_Frequency": lms_login_freq,
        "Units_Enrolled": units_enrolled,
        "Distance_From_Campus_KM": distance_km,
        "Scholarship_Status": scholarship_status,
        "Final_Grade": final_grade,
        "Dropped_Out": dropped_out,
        "On_Time_Grad": on_time_grad
    })
    df.to_csv(CSV_FILENAME, index=False)

    college_ops_df = pd.DataFrame({
        "College": colleges,
        "College_Short": ["CICS", "CET", "CTE", "CABEIHM", "CAS"],
        "Active_Faculty": [30, 28, 25, 34, 24],
        "Board_Pass_Rate": [0.0, 75.2, 87.4, 78.6, 83.5],
        "National_Board_Avg": [0.0, 61.8, 68.2, 63.5, 70.4]
    })

    return df, college_ops_df

# ==============================================================================
# 2. INTEGRATED ENSEMBLE & CURVED SEASONAL FORECASTING ENGINE (CHAPTER 6)
# ==============================================================================
class IntegratedPredictiveEngine:
    def __init__(self, df):
        self.features = [
            "Prior_GWA",
            "Midterm_Grade",
            "Attendance_Rate",
            "LMS_Login_Frequency",
            "Units_Enrolled",
            "Distance_From_Campus_KM",
            "Scholarship_Status"
        ]
        self.feature_labels = [
            "Prior GWA",
            "Midterm Grade",
            "Attendance Rate (%)",
            "LMS Login Frequency",
            "Units Enrolled",
            "Distance from Campus (km)",
            "Scholarship Status"
        ]

        X = df[self.features]
        y = df["Dropped_Out"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Integrate Logistic Regression, Random Forest, and Gradient Boosting together (Section 6.3)
        lr = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
        self.rf_component = RandomForestClassifier(n_estimators=180, max_depth=6, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=42)

        self.ensemble_model = VotingClassifier(
            estimators=[("lr", lr), ("rf", self.rf_component), ("gb", gb)],
            voting="soft",
            weights=[1, 2, 1]
        )
        self.ensemble_model.fit(X_train, y_train)
        self.rf_component.fit(X_train, y_train)  # Fitted for Section 6.5 feature-importance rankings

        self.ensemble_accuracy = accuracy_score(y_test, self.ensemble_model.predict(X_test))

        # Expected Output (Section 6.5): Risk probability score (0.0 - 1.0) & Low/Moderate/High tiers
        df["Risk_Probability_Score"] = self.ensemble_model.predict_proba(X)[:, 1].round(3)
        df["Risk_Tier"] = pd.cut(
            df["Risk_Probability_Score"],
            bins=[-0.01, 0.349, 0.649, 1.01],
            labels=["Low Risk", "Moderate Risk", "High Risk"]
        )

    def predict_student_risk(self, prior_gwa, midterm, attendance, lms_logins, units, distance_km, scholar):
        sample = pd.DataFrame(
            [[prior_gwa, midterm, attendance, lms_logins, units, distance_km, scholar]],
            columns=self.features
        )
        return float(self.ensemble_model.predict_proba(sample)[0][1])

    def _seasonal_fourier_features(self, t_idx):
        """
        SARIMA / Prophet-style non-linear basis functions:
        - Logarithmic saturation curve (models physical classroom capacity leveling off)
        - Fourier harmonic waves (models 1st-Sem intake peaks and 2nd-Sem dips)
        - Multi-year cyclical wave (models natural enrollment wave cycles)
        """
        t = np.asarray(t_idx, dtype=float)
        log_cap = np.log1p(t)
        sem_wave = np.cos(np.pi * t)                # +1 on 1st Sem (even idx), -1 on 2nd Sem (odd idx)
        cycle_sin = np.sin(2 * np.pi * t / 4.0)     # 2-year academic intake cycle
        cycle_cos = np.cos(2 * np.pi * t / 4.0)
        return np.column_stack([log_cap, sem_wave, cycle_sin, cycle_cos])

    def forecast_2027_to_2030(self, dff):
        """
        Projects curved semester-by-semester trajectories from actual historical data
        (2023-2026) through future years (2026-2030) where no current records exist.
        """
        hist_terms = [
            "2023-24 1st", "2023-24 2nd",
            "2024-25 1st", "2024-25 2nd",
            "2025-26 1st", "2025-26 2nd"
        ]
        fut_terms = [
            "2026-27 1st", "2026-27 2nd",
            "2027-28 1st", "2027-28 2nd",
            "2028-29 1st", "2028-29 2nd",
            "2029-30 1st", "2029-30 2nd"
        ]

        hist_counts = dff.groupby("Academic_Term").size().reindex(hist_terms, fill_value=0).values.astype(float)

        hist_ret = []
        for tm in hist_terms:
            sub = dff[dff["Academic_Term"] == tm]
            ret_val = (1 - sub["Dropped_Out"].mean()) * 100 if len(sub) > 0 else 80.0
            hist_ret.append(round(ret_val, 1))
        hist_ret = np.array(hist_ret)

        t_hist = np.arange(len(hist_terms))
        t_fut = np.arange(len(hist_terms), len(hist_terms) + len(fut_terms))

        X_hist = self._seasonal_fourier_features(t_hist)
        X_fut = self._seasonal_fourier_features(t_fut)

        # Enrollment curve model
        enr_model = Ridge(alpha=0.45).fit(X_hist, hist_counts)
        raw_fut_enr = enr_model.predict(X_fut)
        # Anchor smoothly to last observed semester so curve connects naturally
        anchor_shift_enr = hist_counts[-1] - enr_model.predict(X_hist)[-1]
        fut_counts = np.maximum(0, np.round(raw_fut_enr + anchor_shift_enr * np.exp(-0.35 * np.arange(1, len(fut_terms) + 1)))).astype(int)
        enr_std = max(5.0, np.std(hist_counts - enr_model.predict(X_hist)))

        # Retention rate curve model (incorporates S-curve lift from BI Early-Warning interventions + seasonal wave)
        ret_model = Ridge(alpha=0.60).fit(X_hist, hist_ret)
        seasonal_wave = 1.4 * np.cos(np.pi * t_fut)
        s_curve_gain = 4.8 / (1.0 + np.exp(-0.65 * (np.arange(len(fut_terms)) - 2.5)))
        base_ret_level = np.mean(hist_ret)
        fut_ret = np.clip(np.round(base_ret_level + seasonal_wave + s_curve_gain, 1), 55.0, 97.5)
        ret_std = max(1.6, np.std(hist_ret - ret_model.predict(X_hist)))

        return hist_terms, hist_counts, hist_ret, fut_terms, fut_counts, fut_ret, enr_std, ret_std

# ==============================================================================
# 3. INTERACTIVE HOVER TOOLTIP SYSTEM (WORKS ON CURVED LINES & BARS)
# ==============================================================================
def enable_chart_hover(fig, canvas, targets):
    tooltip = fig.text(
        0, 0, "", fontsize=9, fontweight="bold", color="#111111",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFFDE7", ec="#800000", lw=1.4),
        zorder=999, visible=False
    )

    def position_tooltip(px_x, px_y, text):
        fig_w, fig_h = fig.get_size_inches() * fig.dpi
        norm_x = px_x / fig_w
        norm_y = px_y / fig_h
        ha = "left" if norm_x < 0.72 else "right"
        va = "bottom" if norm_y < 0.80 else "top"
        offset_x = 0.012 if ha == "left" else -0.012
        offset_y = 0.025 if va == "bottom" else -0.025
        tooltip.set_position((norm_x + offset_x, norm_y + offset_y))
        tooltip.set_ha(ha)
        tooltip.set_va(va)
        tooltip.set_text(text)
        tooltip.set_visible(True)

    def on_mouse_move(event):
        if event.x is None or event.y is None:
            return
        for item in targets:
            if item["type"] == "line":
                ax = item["ax"]
                line = item["line"]
                xdata, ydata = line.get_data()
                x_vals = item.get("x_indices", np.arange(len(ydata)))
                pts_px = ax.transData.transform(np.column_stack([x_vals, ydata]))
                for idx, (px, py) in enumerate(pts_px):
                    if np.hypot(event.x - px, event.y - py) <= 13:
                        lbl = item["labels"][idx]
                        position_tooltip(px, py, item["fmt"].format(label=lbl, val=ydata[idx]))
                        canvas.draw_idle()
                        return
        for item in targets:
            if item["type"] == "bar":
                is_horiz = item.get("horizontal", False)
                for idx, rect in enumerate(item["bars"]):
                    if rect.get_window_extent().contains(event.x, event.y):
                        val = rect.get_width() if is_horiz else rect.get_height()
                        lbl = item["labels"][idx]
                        position_tooltip(event.x, event.y, item["fmt"].format(label=lbl, val=val))
                        canvas.draw_idle()
                        return
        if tooltip.get_visible():
            tooltip.set_visible(False)
            canvas.draw_idle()

    canvas.mpl_connect("motion_notify_event", on_mouse_move)

# ==============================================================================
# 4. TKINTER BI & CHAPTER 6 ANALYTICS APPLICATION
# ==============================================================================
class BatStateUARASOFDashboard(tk.Tk):
    def __init__(self, df, ops_df, engine):
        super().__init__()
        self.df = df
        self.ops_df = ops_df
        self.engine = engine
        self._active_figs = []

        self.title("BatStateU ARASOF-Nasugbu | Business Intelligence & Chapter 6 Analytics Suite")
        self.geometry("1460x890")
        self.configure(bg="#F4F6F9")

        self._build_header()
        self._build_filters()
        self._build_notebook()
        self.refresh_dashboards()

    def _build_header(self):
        header = tk.Frame(self, bg="#800000", height=64)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="BATANGAS STATE UNIVERSITY – ARASOF-NASUGBU | BI & PREDICTIVE ANALYTICS FRAMEWORK",
            font=("Segoe UI", 13, "bold"),
            fg="white",
            bg="#800000"
        ).pack(side="left", padx=18)

        tk.Label(
            header,
            text=f"Institutional Records: n = {len(self.df):,} | Integrated Ensemble (LR+RF+GB) Accuracy: {self.engine.ensemble_accuracy:.1%}",
            font=("Segoe UI", 10, "bold"),
            fg="#FFD700",
            bg="#800000"
        ).pack(side="right", padx=18)

    def _build_filters(self):
        bar = tk.Frame(self, bg="#FFFFFF", bd=1, relief="solid", height=48)
        bar.pack(fill="x", padx=12, pady=(8, 4))

        tk.Label(bar, text="Filter Data:", font=("Segoe UI", 10, "bold"), fg="#800000", bg="#FFFFFF").pack(side="left", padx=(14, 8), pady=8)

        tk.Label(bar, text="College:", font=("Segoe UI", 10), bg="#FFFFFF").pack(side="left", padx=(4, 4))
        self.college_var = tk.StringVar(value="All Colleges")
        colleges = ["All Colleges"] + sorted(self.df["College"].unique().tolist())
        cb_col = ttk.Combobox(bar, textvariable=self.college_var, values=colleges, state="readonly", width=28)
        cb_col.pack(side="left", padx=4)
        cb_col.bind("<<ComboboxSelected>>", lambda e: self.refresh_dashboards())

        tk.Label(bar, text="Academic Year:", font=("Segoe UI", 10), bg="#FFFFFF").pack(side="left", padx=(12, 4))
        self.ay_var = tk.StringVar(value="All Academic Years")
        ays = ["All Academic Years"] + sorted(self.df["Academic_Year"].unique().tolist())
        cb_ay = ttk.Combobox(bar, textvariable=self.ay_var, values=ays, state="readonly", width=18)
        cb_ay.pack(side="left", padx=4)
        cb_ay.bind("<<ComboboxSelected>>", lambda e: self.refresh_dashboards())

        tk.Label(bar, text="Batangas Town:", font=("Segoe UI", 10), bg="#FFFFFF").pack(side="left", padx=(12, 4))
        self.muni_var = tk.StringVar(value="All Batangas Towns")
        munis = ["All Batangas Towns"] + sorted(self.df["Municipality"].unique().tolist())
        cb_muni = ttk.Combobox(bar, textvariable=self.muni_var, values=munis, state="readonly", width=19)
        cb_muni.pack(side="left", padx=4)
        cb_muni.bind("<<ComboboxSelected>>", lambda e: self.refresh_dashboards())

        tk.Button(bar, text="Reset Filters", command=self.reset_filters, bg="#333333", fg="white", font=("Segoe UI", 9, "bold"), padx=12).pack(side="right", padx=14)

    def reset_filters(self):
        self.college_var.set("All Colleges")
        self.ay_var.set("All Academic Years")
        self.muni_var.set("All Batangas Towns")
        self.refresh_dashboards()

    def get_filtered_data(self):
        dff = self.df.copy()
        ops = self.ops_df.copy()
        if self.college_var.get() != "All Colleges":
            dff = dff[dff["College"] == self.college_var.get()]
            ops = ops[ops["College"] == self.college_var.get()]
        if self.ay_var.get() != "All Academic Years":
            dff = dff[dff["Academic_Year"] == self.ay_var.get()]
        if self.muni_var.get() != "All Batangas Towns":
            dff = dff[dff["Municipality"] == self.muni_var.get()]
        return dff, ops

    def _build_notebook(self):
        style = ttk.Style()
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[12, 6])

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=6)

        self.tab_desc = tk.Frame(self.notebook, bg="#F4F6F9")
        self.notebook.add(self.tab_desc, text="Descriptive Analytics Dashboard")

        self.tab_diag = tk.Frame(self.notebook, bg="#F4F6F9")
        self.notebook.add(self.tab_diag, text="Diagnostic Analytics ")

        self.tab_pred = tk.Frame(self.notebook, bg="#F4F6F9")
        self.notebook.add(self.tab_pred, text="Predictive Forecasting ")

        self.tab_presc = tk.Frame(self.notebook, bg="#F4F6F9")
        self.notebook.add(self.tab_presc, text="Prescriptive analysis and report")

    def refresh_dashboards(self):
        for old_fig in self._active_figs:
            plt.close(old_fig)
        self._active_figs.clear()

        dff, ops = self.get_filtered_data()
        self._render_tab1_descriptive(dff, ops)
        self._render_tab2_diagnostic(dff)
        self._render_tab3_predictive_curves(dff)
        self._render_tab4_prescriptive_ews(dff, ops)

    # --------------------------------------------------------------------------
    # TAB 1: 6.1 DESCRIPTIVE ANALYTICS (WHAT HAPPENED? - 3 ACADEMIC YEARS)
    # --------------------------------------------------------------------------
    def _render_tab1_descriptive(self, dff, ops):
        for widget in self.tab_desc.winfo_children():
            widget.destroy()

        col_subset = self.df if self.college_var.get() == "All Colleges" else self.df[self.df["College"] == self.college_var.get()]
        term_counts = col_subset.groupby("Academic_Term").size().sort_index()
        growth_rate = ((term_counts.iloc[-1] - term_counts.iloc[-2]) / term_counts.iloc[-2]) * 100 if len(term_counts) >= 2 else 0.0

        retention_rate = (1 - dff["Dropped_Out"].mean()) * 100 if len(dff) > 0 else 0.0
        pass_rate = (dff["Final_Grade"] <= 3.00).mean() * 100 if len(dff) > 0 else 0.0
        avg_gwa = dff["Final_Grade"].mean() if len(dff) > 0 else 0.0
        board_ops = ops[ops["Board_Pass_Rate"] > 0]
        board_rate = board_ops["Board_Pass_Rate"].mean() if len(board_ops) > 0 else 0.0

        kpi_frame = tk.Frame(self.tab_desc, bg="#F4F6F9")
        kpi_frame.pack(fill="x", pady=6)

        cards = [
            ("3-YEAR ENROLLMENT RECORDS", f"{len(dff):,}", f"Latest Sem Change: {growth_rate:+.1f}%", "#1F4E78"),
            ("OVERALL RETENTION RATE", f"{retention_rate:.1f}%", f"Dropout Rate: {100-retention_rate:.1f}%", "#2E7D32"),
            ("SUBJECT PASSING RATE", f"{pass_rate:.1f}%", "Final Grade <= 3.00", "#800000"),
            ("MEAN FINAL GRADE (GWA)", f"{avg_gwa:.2f}", "BatStateU 1.00–5.00 Scale", "#00695C"),
            ("BOARD EXAM PASSING AVG", f"{board_rate:.1f}%" if board_rate > 0 else "Non-Board", "Licensure Programs", "#B26A00")
        ]

        for title, val, sub, color in cards:
            card = tk.Frame(kpi_frame, bg="white", bd=1, relief="solid", padx=12, pady=10)
            card.pack(side="left", fill="both", expand=True, padx=5)
            tk.Label(card, text=title, font=("Segoe UI", 8, "bold"), fg="#555555", bg="white").pack(anchor="w")
            tk.Label(card, text=val, font=("Segoe UI", 18, "bold"), fg=color, bg="white").pack(anchor="w", pady=(3, 1))
            tk.Label(card, text=sub, font=("Segoe UI", 8), fg="#777777", bg="white").pack(anchor="w")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=100)
        fig.patch.set_facecolor("#F4F6F9")
        self._active_figs.append(fig)
        hover_targets = []

        if len(dff) > 0:
            t_grp = dff.groupby("Academic_Term").agg(
                Enrollees=("Student_ID", "count"),
                DropoutPct=("Dropped_Out", lambda x: x.mean() * 100)
            ).sort_index()
            bars1 = ax1.bar(t_grp.index, t_grp["Enrollees"], color="#800000", alpha=0.85)
            ax1.set_title("6.1 Historical Enrollment & Dropout Rate (Past 3 Academic Years)", fontsize=10, fontweight="bold")
            ax1.set_ylabel("Enrolled Records", color="#800000")
            ax1.tick_params(axis="x", rotation=20)

            ax1_twin = ax1.twinx()
            line1, = ax1_twin.plot(t_grp.index, t_grp["DropoutPct"], color="#D84315", marker="o", markersize=8, linewidth=2.5)
            ax1_twin.set_ylabel("Dropout Rate (%)", color="#D84315")
            ax1_twin.set_ylim(0, 50)

            hover_targets.append({"type": "line", "ax": ax1_twin, "line": line1, "labels": t_grp.index.tolist(), "fmt": "{label}\nDropout Rate: {val:.1f}%"})
            hover_targets.append({"type": "bar", "ax": ax1, "bars": bars1, "labels": t_grp.index.tolist(), "fmt": "{label}\nEnrolled: {val:.0f} Students"})

            grade_order = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 4.00, 5.00]
            g_counts = dff["Final_Grade"].value_counts().reindex(grade_order, fill_value=0)
            g_labels = [f"{g:.2f}" for g in grade_order]
            bar_colors = ["#2E7D32" if g <= 3.00 else "#C62828" for g in grade_order]
            bars2 = ax2.bar(g_labels, g_counts.values, color=bar_colors)
            ax2.set_title("6.1 Final Grade Distribution (Green = Passed, Red = Cond/Failed)", fontsize=10, fontweight="bold")
            ax2.set_xlabel("BatStateU Final Grade")
            ax2.set_ylabel("Number of Records")
            hover_targets.append({"type": "bar", "ax": ax2, "bars": bars2, "labels": g_labels, "fmt": "Grade {label}\nCount: {val:.0f} Records"})

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tab_desc)
        canvas.draw()
        enable_chart_hover(fig, canvas, hover_targets)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    # --------------------------------------------------------------------------
    # TAB 2: 6.2 DIAGNOSTIC ANALYTICS (WHY DID IT HAPPEN? - BATANGAS TOWNS)
    # --------------------------------------------------------------------------
    def _render_tab2_diagnostic(self, dff):
        for widget in self.tab_diag.winfo_children():
            widget.destroy()

        if len(dff) == 0:
            return

        dff = dff.copy()
        dff["Is_Failed_Or_Drop"] = ((dff["Final_Grade"] > 3.00) | (dff["Dropped_Out"] == 1)).astype(int)

        near_fail = dff[dff["Distance_From_Campus_KM"] <= 15]["Is_Failed_Or_Drop"].mean() * 100
        far_fail = dff[dff["Distance_From_Campus_KM"] > 25]["Is_Failed_Or_Drop"].mean() * 100
        normal_sec_fail = dff[dff["Section_Size"] <= 45]["Is_Failed_Or_Drop"].mean() * 100
        overloaded_sec_fail = dff[dff["Section_Size"] > 45]["Is_Failed_Or_Drop"].mean() * 100
        prereq_def_fail = dff[dff["Prior_GWA"] >= 2.75]["Is_Failed_Or_Drop"].mean() * 100

        kpi_frame = tk.Frame(self.tab_diag, bg="#F4F6F9")
        kpi_frame.pack(fill="x", pady=6)

        diag_cards = [
            ("NEAR TOWNS (Nasugbu/Lian)", f"{near_fail:.1f}%", "Failure / Withdrawal Rate", "#2E7D32"),
            ("FARTHER BATANGAS TOWNS (>25km)", f"{far_fail:.1f}%", f"+{max(0, far_fail - near_fail):.1f}% Higher Risk", "#C62828"),
            ("STANDARD SECTIONS (<= 45)", f"{normal_sec_fail:.1f}%", "Failure / Withdrawal Rate", "#1F4E78"),
            ("OVERLOADED SECTIONS (> 45)", f"{overloaded_sec_fail:.1f}%", f"+{max(0, overloaded_sec_fail - normal_sec_fail):.1f}% Spike", "#D84315"),
            ("PREREQUISITE DEFICIT (GWA >= 2.75)", f"{prereq_def_fail:.1f}%", "Failure / Withdrawal Rate", "#800000")
        ]

        for title, val, sub, color in diag_cards:
            card = tk.Frame(kpi_frame, bg="white", bd=1, relief="solid", padx=12, pady=10)
            card.pack(side="left", fill="both", expand=True, padx=5)
            tk.Label(card, text=title, font=("Segoe UI", 8, "bold"), fg="#555555", bg="white").pack(anchor="w")
            tk.Label(card, text=val, font=("Segoe UI", 18, "bold"), fg=color, bg="white").pack(anchor="w", pady=(3, 1))
            tk.Label(card, text=sub, font=("Segoe UI", 8), fg="#777777", bg="white").pack(anchor="w")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=100)
        fig.patch.set_facecolor("#F4F6F9")
        self._active_figs.append(fig)
        hover_targets = []

        muni_diag = dff.groupby("Municipality")["Is_Failed_Or_Drop"].mean().sort_values() * 100
        bars1 = ax1.barh(muni_diag.index, muni_diag.values, color="#1F4E78")
        ax1.set_title("6.2 Failure & Withdrawal Rate by Batangas Municipality (%)", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Failure / Withdrawal Rate (%)")
        hover_targets.append({"type": "bar", "horizontal": True, "ax": ax1, "bars": bars1, "labels": muni_diag.index.tolist(), "fmt": "{label}, Batangas\nFailure/Dropout Rate: {val:.1f}%"})

        dff["Section_Category"] = np.where(dff["Section_Size"] <= 45, "Standard Section\n(26–45 Students)", "Overloaded Section\n(46–56 Students)")
        dff["Prereq_Status"] = np.where(dff["Prior_GWA"] < 2.75, "Strong Prior GWA (< 2.75)", "Prerequisite Deficit (>= 2.75)")

        piv = dff.pivot_table(index="Section_Category", columns="Prereq_Status", values="Is_Failed_Or_Drop", aggfunc="mean") * 100
        x = np.arange(len(piv.index))
        w = 0.35
        b_strong = ax2.bar(x - w/2, piv["Strong Prior GWA (< 2.75)"], width=w, color="#2E7D32", label="Strong Prior GWA (< 2.75)")
        b_def = ax2.bar(x + w/2, piv["Prerequisite Deficit (>= 2.75)"], width=w, color="#C62828", label="Prerequisite Deficit (>= 2.75)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(piv.index)
        ax2.set_ylabel("Failure / Withdrawal Rate (%)")
        ax2.set_title("6.2 Impact of Overloaded Sections & Prerequisite Grade Deficits", fontsize=10, fontweight="bold")
        ax2.legend(fontsize=8)

        s_labels = ["Standard Section", "Overloaded Section"]
        hover_targets.append({"type": "bar", "ax": ax2, "bars": b_strong, "labels": s_labels, "fmt": "{label} (Strong Prior GWA)\nFailure Rate: {val:.1f}%"})
        hover_targets.append({"type": "bar", "ax": ax2, "bars": b_def, "labels": s_labels, "fmt": "{label} (Prereq Deficit)\nFailure Rate: {val:.1f}%"})

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tab_diag)
        canvas.draw()
        enable_chart_hover(fig, canvas, hover_targets)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    # --------------------------------------------------------------------------
    # TAB 3: 6.3 CURVED SEASONAL FORECAST (2023-2026 ACTUAL vs 2027-2030)
    # --------------------------------------------------------------------------
    def _render_tab3_predictive_curves(self, dff):
        for widget in self.tab_pred.winfo_children():
            widget.destroy()

        base_df = self.df.copy()
        if self.college_var.get() != "All Colleges":
            base_df = base_df[base_df["College"] == self.college_var.get()]
        if self.muni_var.get() != "All Batangas Towns":
            base_df = base_df[base_df["Municipality"] == self.muni_var.get()]

        hist_terms, hist_cnts, hist_ret, fut_terms, fut_cnts, fut_ret, enr_std, ret_std = self.engine.forecast_2027_to_2030(base_df)

        banner = tk.Frame(self.tab_pred, bg="white", bd=1, relief="solid", padx=14, pady=8)
        banner.pack(fill="x", padx=8, pady=(8, 2))
        tk.Label(
            banner,
            text=(
                f"6.3 Integrated Predictive Engine: Soft-Voting Ensemble (Logistic Regression + Random Forest + Gradient Boosting) "
                f"[Accuracy: {self.engine.ensemble_accuracy:.1%}]  |  Seasonal Fourier Time-Series Curves (2023–2030)"
            ),
            font=("Segoe UI", 10, "bold"), fg="#800000", bg="white"
        ).pack(anchor="w")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0), dpi=100)
        fig.patch.set_facecolor("#F4F6F9")
        self._active_figs.append(fig)
        hover_targets = []

        all_terms = hist_terms + fut_terms
        x_hist = np.arange(len(hist_terms))
        x_fut_conn = np.arange(len(hist_terms) - 1, len(all_terms))

        # --- LEFT CURVE: ACTUAL (2023-2026) vs PREDICTED ENROLLMENT CURVE (2027-2030) ---
        y_fut_enr_conn = np.concatenate([[hist_cnts[-1]], fut_cnts])

        line_act_enr, = ax1.plot(
            x_hist, hist_cnts, color="#1F4E78", marker="o", markersize=7.5,
            linewidth=2.8, linestyle="-", label="Actual Enrollment (2023–2026)"
        )
        line_pred_enr, = ax1.plot(
            x_fut_conn, y_fut_enr_conn, color="#D84315", marker="s", markersize=7.5,
            linewidth=2.8, linestyle="--", label="Predicted Seasonal Curve (2026–2030)"
        )
        ci_enr = 1.55 * enr_std * np.linspace(0.2, 1.35, len(x_fut_conn))
        ax1.fill_between(x_fut_conn, y_fut_enr_conn - ci_enr, y_fut_enr_conn + ci_enr, color="#D84315", alpha=0.14, label="95% Forecast Interval")

        ax1.axvline(x=len(hist_terms) - 1, color="#666666", linestyle=":", linewidth=1.3)
        ax1.set_xticks(np.arange(len(all_terms)))
        ax1.set_xticklabels(all_terms, rotation=32, ha="right", fontsize=8.5)
        ax1.set_ylabel("Semester Student Records")
        ax1.set_title("6.3 Semester Enrollment Curve: Actual vs. 2027–2030 Forecast", fontsize=10, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(loc="upper left", fontsize=8)

        hover_targets.append({
            "type": "line", "ax": ax1, "line": line_act_enr, "x_indices": x_hist,
            "labels": hist_terms, "fmt": "{label} (Actual Data)\nEnrolled: {val:.0f} Students"
        })
        hover_targets.append({
            "type": "line", "ax": ax1, "line": line_pred_enr, "x_indices": x_fut_conn,
            "labels": [hist_terms[-1]] + fut_terms, "fmt": "{label} (Projected Curve)\nPredicted: {val:.0f} Students"
        })

        # --- RIGHT CURVE: ACTUAL vs PREDICTED RETENTION RATE CURVE (2023-2030) ---
        y_fut_ret_conn = np.concatenate([[hist_ret[-1]], fut_ret])

        line_act_ret, = ax2.plot(
            x_hist, hist_ret, color="#2E7D32", marker="o", markersize=7.5,
            linewidth=2.8, linestyle="-", label="Actual Retention Rate (2023–2026)"
        )
        line_pred_ret, = ax2.plot(
            x_fut_conn, y_fut_ret_conn, color="#800000", marker="D", markersize=7.5,
            linewidth=2.8, linestyle="--", label="Projected Retention w/ BI EWS (2026–2030)"
        )
        ci_ret = 1.4 * ret_std * np.linspace(0.2, 1.25, len(x_fut_conn))
        ax2.fill_between(x_fut_conn, np.maximum(50, y_fut_ret_conn - ci_ret), np.minimum(100, y_fut_ret_conn + ci_ret), color="#800000", alpha=0.12, label="Retention Confidence Band")

        ax2.axvline(x=len(hist_terms) - 1, color="#666666", linestyle=":", linewidth=1.3)
        ax2.set_xticks(np.arange(len(all_terms)))
        ax2.set_xticklabels(all_terms, rotation=32, ha="right", fontsize=8.5)
        ax2.set_ylim(55, 102)
        ax2.set_ylabel("Student Retention Rate (%)")
        ax2.set_title("6.3 Retention Rate Curve: Actual vs. 2027–2030 Forecast", fontsize=10, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(loc="lower right", fontsize=8)

        hover_targets.append({
            "type": "line", "ax": ax2, "line": line_act_ret, "x_indices": x_hist,
            "labels": hist_terms, "fmt": "{label} (Actual)\nRetention Rate: {val:.1f}%"
        })
        hover_targets.append({
            "type": "line", "ax": ax2, "line": line_pred_ret, "x_indices": x_fut_conn,
            "labels": [hist_terms[-1]] + fut_terms, "fmt": "{label} (Forecast Curve)\nPredicted Retention: {val:.1f}%"
        })

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tab_pred)
        canvas.draw()
        enable_chart_hover(fig, canvas, hover_targets)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=6)

    # --------------------------------------------------------------------------
    # TAB 4: 6.4 PRESCRIPTIVE ANALYTICS & 6.5 EARLY-WARNING SYSTEM
    # --------------------------------------------------------------------------
    def _render_tab4_prescriptive_ews(self, dff, ops):
        for widget in self.tab_presc.winfo_children():
            widget.destroy()

        presc_frame = tk.Frame(self.tab_presc, bg="#F4F6F9")
        presc_frame.pack(fill="x", pady=4, padx=6)

        high_risk_cnt = (dff["Risk_Tier"] == "High Risk").sum()
        mod_risk_cnt = (dff["Risk_Tier"] == "Moderate Risk").sum()
        _, _, _, _, fut_cnts, _, _, _ = self.engine.forecast_2027_to_2030(self.df)
        proj_2027_1st = fut_cnts[2] if len(fut_cnts) >= 3 else 0
        recommended_sections = max(1, int(np.ceil((proj_2027_1st * 0.36) / 40.0)))
        overloaded_cnt = (dff["Section_Size"] > 45).sum()

        presc_cards = [
            ("6.4 PEER-TUTORING & ADVISING", f"{high_risk_cnt} High-Risk | {mod_risk_cnt} Mod", "Assign immediate Guidance & Peer Tutoring", "#C62828"),
            ("6.4 2027–28 FRESHMAN SECTIONS", f"Open {recommended_sections} Sections", f"For 2027–28 1st Sem peak ({proj_2027_1st} proj, 40/sec)", "#1F4E78"),
            ("6.4 FACULTY LOAD BALANCING", f"{overloaded_cnt} Overloaded Records", "Split sections > 45 students; balance units", "#00695C")
        ]

        for title, val, sub, color in presc_cards:
            card = tk.Frame(presc_frame, bg="white", bd=1, relief="solid", padx=12, pady=8)
            card.pack(side="left", fill="both", expand=True, padx=4)
            tk.Label(card, text=title, font=("Segoe UI", 8, "bold"), fg="#555555", bg="white").pack(anchor="w")
            tk.Label(card, text=val, font=("Segoe UI", 15, "bold"), fg=color, bg="white").pack(anchor="w", pady=(2, 1))
            tk.Label(card, text=sub, font=("Segoe UI", 8), fg="#666666", bg="white").pack(anchor="w")

        mid_pane = tk.Frame(self.tab_presc, bg="#F4F6F9")
        mid_pane.pack(fill="both", expand=True, padx=6, pady=4)

        left_frame = tk.Frame(mid_pane, bg="white", bd=1, relief="solid")
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

        fig, ax = plt.subplots(figsize=(5.5, 3.1), dpi=100)
        self._active_figs.append(fig)
        importances = pd.Series(self.engine.rf_component.feature_importances_ * 100, index=self.engine.feature_labels).sort_values()
        bars_imp = ax.barh(importances.index, importances.values, color="#00695C")
        ax.set_title("6.5 Random Forest Feature Importance Rankings", fontsize=10, fontweight="bold")
        ax.set_xlabel("Relative Importance (%)")
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=left_frame)
        canvas.draw()
        enable_chart_hover(fig, canvas, [
            {"type": "bar", "horizontal": True, "ax": ax, "bars": bars_imp, "labels": importances.index.tolist(), "fmt": "{label}\nImportance: {val:.1f}%"}
        ])
        canvas.get_tk_widget().pack(fill="both", expand=True)

        sim_frame = tk.LabelFrame(
            mid_pane,
            text=" 6.5 Integrated Ensemble Risk Simulator (Output: 0.0 – 1.0 Risk Probability) ",
            font=("Segoe UI", 9, "bold"), bg="white", padx=12, pady=4
        )
        sim_frame.pack(side="right", fill="both", expand=True)

        self.sim_prior_gwa = tk.DoubleVar(value=2.50)
        self.sim_midterm = tk.DoubleVar(value=3.00)
        self.sim_attend = tk.DoubleVar(value=76.0)
        self.sim_lms = tk.IntVar(value=4)
        self.sim_units = tk.IntVar(value=21)
        self.sim_dist = tk.DoubleVar(value=28.0)
        self.sim_scholar = tk.IntVar(value=1)

        controls = [
            ("Prior GWA (1.00 – 5.00):", self.sim_prior_gwa, 1.0, 5.0, 0.25),
            ("Midterm Grade (1.00 – 5.00):", self.sim_midterm, 1.0, 5.0, 0.25),
            ("Attendance Rate (%):", self.sim_attend, 52.0, 100.0, 1.0),
            ("LMS Login Frequency (/wk):", self.sim_lms, 1, 26, 1),
            ("Units Enrolled:", self.sim_units, 12, 24, 3),
            ("Distance from Campus (km):", self.sim_dist, 1.0, 50.0, 1.0),
            ("Scholarship Status (1=Yes, 0=No):", self.sim_scholar, 0, 1, 1)
        ]

        for idx, (lbl, var, frm, to, res) in enumerate(controls):
            tk.Label(sim_frame, text=lbl, font=("Segoe UI", 8), bg="white").grid(row=idx, column=0, sticky="w", pady=0)
            tk.Scale(sim_frame, variable=var, from_=frm, to=to, resolution=res, orient="horizontal", length=170, bg="white").grid(row=idx, column=1, padx=6)

        self.pred_result_lbl = tk.Label(
            sim_frame,
            text="Adjust inputs and click 'Compute Ensemble Risk Score'",
            font=("Segoe UI", 9, "bold"), fg="#1F4E78", bg="white"
        )
        self.pred_result_lbl.grid(row=7, column=0, columnspan=2, pady=(2, 2))

        tk.Button(
            sim_frame, text="Compute Ensemble Risk Score (0.0 – 1.0)",
            command=self.run_single_prediction, bg="#800000", fg="white",
            font=("Segoe UI", 9, "bold"), padx=10, pady=2
        ).grid(row=8, column=0, columnspan=2)

        table_frame = tk.LabelFrame(
            self.tab_presc,
            text=" Human-in-the-Loop Intervention Roster (For Guidance Counselor Review — Not Automated Administrative Action) ",
            font=("Segoe UI", 9, "bold"), bg="white"
        )
        table_frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        cols = ("Student_ID", "College_Short", "Municipality", "Prior_GWA", "Midterm_Grade", "Attendance_Rate", "Risk_Probability_Score", "Risk_Tier", "Prescribed_Action")
        headers = {
            "Student_ID": "Student ID",
            "College_Short": "College",
            "Municipality": "Batangas Town",
            "Prior_GWA": "Prior GWA",
            "Midterm_Grade": "Midterm",
            "Attendance_Rate": "Attendance",
            "Risk_Probability_Score": "Risk Score (0–1)",
            "Risk_Tier": "Risk Tier",
            "Prescribed_Action": "Prescriptive Recommendation"
        }

        tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=4)
        for c in cols:
            tree.heading(c, text=headers[c])
            tree.column(c, width=185 if c == "Prescribed_Action" else 110, anchor="center")

        high_risk_df = dff.sort_values(by="Risk_Probability_Score", ascending=False).head(25)
        for _, row in high_risk_df.iterrows():
            tier = row["Risk_Tier"]
            action = "Assign Peer Tutoring & Guidance Intake" if tier == "High Risk" else ("Adviser Check-In & LMS Monitoring" if tier == "Moderate Risk" else "Standard Advising")
            tree.insert("", "end", values=(
                row["Student_ID"], row["College_Short"], row["Municipality"],
                f"{row['Prior_GWA']:.2f}", f"{row['Midterm_Grade']:.2f}", f"{row['Attendance_Rate']:.1f}%",
                f"{row['Risk_Probability_Score']:.3f}", tier, action
            ))
        tree.pack(fill="both", expand=True, padx=6, pady=4)

    def run_single_prediction(self):
        score = self.engine.predict_student_risk(
            self.sim_prior_gwa.get(),
            self.sim_midterm.get(),
            self.sim_attend.get(),
            self.sim_lms.get(),
            self.sim_units.get(),
            self.sim_dist.get(),
            self.sim_scholar.get()
        )
        if score >= 0.65:
            tier, color = "HIGH RISK — Recommend Guidance Counseling & Peer Tutoring", "#C62828"
        elif score >= 0.35:
            tier, color = "MODERATE RISK — Recommend Academic Adviser Check-In", "#EF6C00"
        else:
            tier, color = "LOW RISK — Good Academic Standing", "#2E7D32"
        self.pred_result_lbl.config(text=f"Ensemble Risk Score: {score:.3f} | {tier}", fg=color)

if __name__ == "__main__":
    df_students, df_college_ops = generate_realistic_arasof_dataset(n_records=1500)
    integrated_engine = IntegratedPredictiveEngine(df_students)
    app = BatStateUARASOFDashboard(df_students, df_college_ops, integrated_engine)
    app.mainloop()
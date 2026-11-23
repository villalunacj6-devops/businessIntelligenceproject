import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==============================================================================
# 1. BATSTATEU ARASOF-NASUGBU DATASET SIMULATION (1,000 STUDENT RECORDS)
# ==============================================================================
CSV_FILENAME = "batstateu_arasof_batangas_1000.csv"


def generate_arasof_dataset(n_records=1000, seed=42):
    np.random.seed(seed)

    colleges = ["CICS", "CABEIHM", "CET", "CTE", "CAS", "CNAHS"]
    college_probs = [0.22, 0.23, 0.18, 0.16, 0.12, 0.09]

    programs_map = {
        "CICS": ["BS Information Technology", "BS Computer Science"],
        "CABEIHM": ["BS Accountancy", "BS Business Administration", "BS Hospitality Management",
                    "BS Tourism Management"],
        "CET": ["BS Computer Engineering Tech", "BS Electronics Engineering Tech", "BS Electrical Engineering Tech"],
        "CTE": ["Bachelor of Secondary Education", "Bachelor of Elementary Education"],
        "CAS": ["BS Fisheries and Aquatic Sciences", "BS Food Technology", "BS Psychology", "BA Communication"],
        "CNAHS": ["BS Nursing", "BS Nutrition and Dietetics"]
    }

    municipalities = [
        "Nasugbu", "Lian", "Tuy", "Balayan",
        "Calatagan", "Lemery", "Calaca", "Taal"
    ]
    muni_probs = [0.42, 0.18, 0.13, 0.11, 0.07, 0.04, 0.03, 0.02]

    travel_base = {
        "Nasugbu": 15, "Lian": 25, "Tuy": 40, "Balayan": 55,
        "Calatagan": 65, "Lemery": 75, "Calaca": 65, "Taal": 80
    }

    terms = ["2023-2024 1st", "2023-2024 2nd", "2024-2025 1st", "2024-2025 2nd", "2025-2026 1st"]
    term_probs = [0.16, 0.17, 0.20, 0.21, 0.26]

    chosen_colleges = np.random.choice(colleges, size=n_records, p=college_probs)
    chosen_programs = [np.random.choice(programs_map[c]) for c in chosen_colleges]
    chosen_terms = np.random.choice(terms, size=n_records, p=term_probs)
    year_levels = np.random.choice([1, 2, 3, 4], size=n_records, p=[0.35, 0.26, 0.21, 0.18])
    chosen_muni = np.random.choice(municipalities, size=n_records, p=muni_probs)

    travel_mins = np.array([
        max(10, int(np.random.normal(travel_base[m], 7))) for m in chosen_muni
    ])

    cohort_prefix = np.where(year_levels == 1, "25",
                             np.where(year_levels == 2, "24", np.where(year_levels == 3, "23", "22")))
    student_ids = [f"{cohort_prefix[i]}-7{1000 + i:04d}" for i in range(n_records)]

    has_scholarship = np.random.choice([1, 0], size=n_records, p=[0.85, 0.15])

    attendance_rate = np.clip(
        np.random.normal(92.0, 7.5, size=n_records) - (travel_mins * 0.11),
        50.0, 100.0
    ).round(1)

    portal_logins = np.clip(
        np.random.poisson(10, size=n_records) - (travel_mins // 30),
        0, 25
    )

    batstateu_grades = np.array([1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 4.00, 5.00])

    def nearest_grade(raw_score):
        return batstateu_grades[np.abs(batstateu_grades - raw_score).argmin()]

    raw_midterm = (
            3.60
            - (attendance_rate / 100.0) * 1.55
            - (portal_logins / 25.0) * 0.50
            + np.random.normal(0, 0.30, size=n_records)
    )
    midterm_grade = np.array([nearest_grade(g) for g in raw_midterm])

    raw_final = midterm_grade + np.random.normal(-0.05, 0.20, size=n_records)
    final_grade = np.array([nearest_grade(g) for g in raw_final])

    risk_score = (
            1.85 * (midterm_grade - 2.60)
            - 0.065 * (attendance_rate - 82)
            - 0.11 * portal_logins
            + 0.014 * (travel_mins - 35)
            - 0.55 * has_scholarship
    )
    prob_dropout = 1 / (1 + np.exp(-risk_score))
    dropped_out = (np.random.rand(n_records) < prob_dropout).astype(int)

    on_time_grad = np.where((year_levels == 4) & (dropped_out == 0) & (final_grade <= 3.00), 1, 0)

    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Academic_Term": chosen_terms,
        "College": chosen_colleges,
        "Program": chosen_programs,
        "Year_Level": year_levels,
        "Municipality": chosen_muni,
        "Travel_Mins": travel_mins,
        "Has_Scholarship": has_scholarship,
        "Attendance_Rate": attendance_rate,
        "Weekly_Portal_Logins": portal_logins,
        "Midterm_Grade": midterm_grade,
        "Final_Grade": final_grade,
        "Dropped_Out": dropped_out,
        "On_Time_Grad": on_time_grad
    })
    df.to_csv(CSV_FILENAME, index=False)

    college_ops_df = pd.DataFrame({
        "College": ["CICS", "CABEIHM", "CET", "CTE", "CAS", "CNAHS"],
        "Active_Faculty": [30, 34, 28, 25, 24, 19],
        "Board_Pass_Rate": [0.0, 78.6, 75.2, 87.4, 83.5, 96.2],
        "National_Board_Avg": [0.0, 63.5, 61.8, 68.2, 70.4, 83.1],
        "Faculty_Load_Compliance": [96.7, 94.1, 93.5, 96.0, 95.8, 97.4],
        "Scholarship_Release_Rate": [95.2, 94.0, 93.4, 95.8, 92.6, 96.5],
        "Budget_Used_Rate": [90.2, 91.5, 92.0, 89.1, 88.4, 93.6],
        "Old_Report_Hours": [38.0, 42.0, 40.0, 32.0, 30.0, 28.0],
        "New_BI_Report_Hours": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
    })

    return df, college_ops_df


# ==============================================================================
# 2. STUDENT DROPOUT PREDICTION ENGINE
# ==============================================================================
class DropoutPredictor:
    def __init__(self, df):
        self.features = [
            "Midterm_Grade", "Attendance_Rate", "Weekly_Portal_Logins",
            "Travel_Mins", "Year_Level", "Has_Scholarship"
        ]
        self.friendly_names = [
            "Midterm Grade", "Class Attendance (%)", "Weekly Portal Activity",
            "Travel Time (Mins)", "Year Level", "Scholarship Status"
        ]
        X = df[self.features]
        y = df["Dropped_Out"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
        self.model.fit(X_train, y_train)

        self.accuracy = accuracy_score(y_test, self.model.predict(X_test))
        df["Dropout_Risk_Percent"] = (self.model.predict_proba(X)[:, 1] * 100).round(1)

    def predict_student(self, midterm, attendance, logins, travel, year_lvl, scholar):
        sample = pd.DataFrame(
            [[midterm, attendance, logins, travel, year_lvl, scholar]],
            columns=self.features
        )
        return self.model.predict_proba(sample)[0][1] * 100


# ==============================================================================
# 3. RELIABLE PIXEL-ACCURATE HOVER TOOLTIP SYSTEM
# ==============================================================================
def enable_chart_hover(fig, canvas, targets):
    """
    Reliable mouse-hover tooltips using direct display-coordinate hit detection.
    Works across dual-axis (twinx), vertical bars, horizontal bars, and line points.
    """
    # Place a single figure-level tooltip annotation so it floats above all axes
    tooltip = fig.text(
        0, 0, "", fontsize=9.5, fontweight="bold", color="#111111",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFFDE7", ec="#800000", lw=1.5),
        zorder=999, visible=False
    )

    def position_tooltip(px_x, px_y, text):
        fig_w, fig_h = fig.get_size_inches() * fig.dpi
        norm_x = px_x / fig_w
        norm_y = px_y / fig_h

        # Keep tooltip inside figure boundaries
        ha = "left" if norm_x < 0.72 else "right"
        va = "bottom" if norm_y < 0.82 else "top"
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

        # 1. Check line markers first (higher precision)
        for item in targets:
            if item["type"] == "line":
                ax = item["ax"]
                line = item["line"]
                xdata, ydata = line.get_data()
                # Convert data points to screen pixels
                pts_px = ax.transData.transform(np.column_stack([np.arange(len(ydata)), ydata]))
                for idx, (px, py) in enumerate(pts_px):
                    if np.hypot(event.x - px, event.y - py) <= 12:
                        lbl = item["labels"][idx]
                        msg = item["fmt"].format(label=lbl, val=ydata[idx])
                        position_tooltip(px, py, msg)
                        canvas.draw_idle()
                        return

        # 2. Check bar rectangles (works even underneath a twinx axis)
        for item in targets:
            if item["type"] == "bar":
                is_horiz = item.get("horizontal", False)
                for idx, rect in enumerate(item["bars"]):
                    bbox = rect.get_window_extent()
                    if bbox.contains(event.x, event.y):
                        val = rect.get_width() if is_horiz else rect.get_height()
                        lbl = item["labels"][idx]
                        msg = item["fmt"].format(label=lbl, val=val)
                        position_tooltip(event.x, event.y, msg)
                        canvas.draw_idle()
                        return

        if tooltip.get_visible():
            tooltip.set_visible(False)
            canvas.draw_idle()

    canvas.mpl_connect("motion_notify_event", on_mouse_move)


# ==============================================================================
# 4. EASY-TO-USE CAMPUS DASHBOARD APPLICATION
# ==============================================================================
class BatStateUARASOFDashboard(tk.Tk):
    def __init__(self, df, ops_df, predictor):
        super().__init__()
        self.df = df
        self.ops_df = ops_df
        self.predictor = predictor
        self._active_figs = []  # Keep references alive so hover events never disconnect

        self.title("BatStateU ARASOF-Nasugbu | Campus Performance & Student Success Dashboard")
        self.geometry("1400x850")
        self.configure(bg="#F4F6F9")

        self._build_header()
        self._build_filters()
        self._build_tabs()
        self.refresh_all()

    def _build_header(self):
        header = tk.Frame(self, bg="#800000", height=64)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="BATANGAS STATE UNIVERSITY – ARASOF-NASUGBU | CAMPUS DECISION DASHBOARD",
            font=("Segoe UI", 13, "bold"),
            fg="white",
            bg="#800000"
        ).pack(side="left", padx=18)

        tk.Label(
            header,
            text=f"Sample Data: {len(self.df):,} Batangas Students | Prediction Accuracy: {self.predictor.accuracy:.1%}",
            font=("Segoe UI", 10, "bold"),
            fg="#FFD700",
            bg="#800000"
        ).pack(side="right", padx=18)

    def _build_filters(self):
        bar = tk.Frame(self, bg="#FFFFFF", bd=1, relief="solid", height=48)
        bar.pack(fill="x", padx=12, pady=(8, 4))

        tk.Label(bar, text="Filter Results by:", font=("Segoe UI", 10, "bold"), fg="#800000", bg="#FFFFFF").pack(
            side="left", padx=(14, 10), pady=8)

        tk.Label(bar, text="College:", font=("Segoe UI", 10), bg="#FFFFFF").pack(side="left", padx=(6, 4))
        self.college_var = tk.StringVar(value="All Colleges")
        colleges = ["All Colleges"] + sorted(self.df["College"].unique().tolist())
        cb_col = ttk.Combobox(bar, textvariable=self.college_var, values=colleges, state="readonly", width=15)
        cb_col.pack(side="left", padx=4)
        cb_col.bind("<<ComboboxSelected>>", lambda e: self.refresh_all())

        tk.Label(bar, text="Semester:", font=("Segoe UI", 10), bg="#FFFFFF").pack(side="left", padx=(14, 4))
        self.term_var = tk.StringVar(value="All Semesters")
        terms = ["All Semesters"] + sorted(self.df["Academic_Term"].unique().tolist())
        cb_term = ttk.Combobox(bar, textvariable=self.term_var, values=terms, state="readonly", width=16)
        cb_term.pack(side="left", padx=4)
        cb_term.bind("<<ComboboxSelected>>", lambda e: self.refresh_all())

        tk.Label(bar, text="Batangas Municipality:", font=("Segoe UI", 10), bg="#FFFFFF").pack(side="left",
                                                                                               padx=(14, 4))
        self.muni_var = tk.StringVar(value="All Towns")
        munis = ["All Towns"] + sorted(self.df["Municipality"].unique().tolist())
        cb_muni = ttk.Combobox(bar, textvariable=self.muni_var, values=munis, state="readonly", width=16)
        cb_muni.pack(side="left", padx=4)
        cb_muni.bind("<<ComboboxSelected>>", lambda e: self.refresh_all())

        tk.Button(bar, text="Show All Data", command=self.reset_filters, bg="#333333", fg="white",
                  font=("Segoe UI", 9, "bold"), padx=12).pack(side="right", padx=14)

    def reset_filters(self):
        self.college_var.set("All Colleges")
        self.term_var.set("All Semesters")
        self.muni_var.set("All Towns")
        self.refresh_all()

    def get_filtered_data(self):
        dff = self.df.copy()
        ops = self.ops_df.copy()
        if self.college_var.get() != "All Colleges":
            dff = dff[dff["College"] == self.college_var.get()]
            ops = ops[ops["College"] == self.college_var.get()]
        if self.term_var.get() != "All Semesters":
            dff = dff[dff["Academic_Term"] == self.term_var.get()]
        if self.muni_var.get() != "All Towns":
            dff = dff[dff["Municipality"] == self.muni_var.get()]
        return dff, ops

    def _build_tabs(self):
        style = ttk.Style()
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[14, 6])

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=6)

        self.tab_exec = tk.Frame(self.notebook, bg="#F4F6F9")
        self.notebook.add(self.tab_exec, text="1. Enrollment, Passing & Board Exams")

        self.tab_ops = tk.Frame(self.notebook, bg="#F4F6F9")
        self.notebook.add(self.tab_ops, text="2. Batangas Towns, Faculty & Budget")

        self.tab_pred = tk.Frame(self.notebook, bg="#F4F6F9")
        self.notebook.add(self.tab_pred, text="3. Student Dropout Predictor (Early Warning)")

    def refresh_all(self):
        for old_fig in self._active_figs:
            plt.close(old_fig)
        self._active_figs.clear()

        dff, ops = self.get_filtered_data()
        self._render_tab1_academic(dff, ops)
        self._render_tab2_operations(dff, ops)
        self._render_tab3_prediction(dff)

    # --------------------------------------------------------------------------
    # TAB 1: ENROLLMENT, PASSING & BOARD EXAMS
    # --------------------------------------------------------------------------
    def _render_tab1_academic(self, dff, ops):
        for widget in self.tab_exec.winfo_children():
            widget.destroy()

        col_subset = self.df if self.college_var.get() == "All Colleges" else self.df[
            self.df["College"] == self.college_var.get()]
        term_counts = col_subset.groupby("Academic_Term").size().sort_index()
        growth_rate = ((term_counts.iloc[-1] - term_counts.iloc[-2]) / term_counts.iloc[-2]) * 100 if len(
            term_counts) >= 2 else 0.0

        yr1 = dff[dff["Year_Level"] == 1]
        retention_rate = (1 - yr1["Dropped_Out"].mean()) * 100 if len(yr1) > 0 else 0.0

        yr4 = dff[dff["Year_Level"] == 4]
        grad_rate = (yr4["On_Time_Grad"].mean()) * 100 if len(yr4) > 0 else 0.0

        pass_rate = (dff["Final_Grade"] <= 3.00).mean() * 100 if len(dff) > 0 else 0.0

        board_ops = ops[ops["Board_Pass_Rate"] > 0]
        board_rate = board_ops["Board_Pass_Rate"].mean() if len(board_ops) > 0 else 0.0
        nat_avg = board_ops["National_Board_Avg"].mean() if len(board_ops) > 0 else 0.0

        kpi_frame = tk.Frame(self.tab_exec, bg="#F4F6F9")
        kpi_frame.pack(fill="x", pady=6)

        cards = [
            ("ENROLLMENT GROWTH", f"{growth_rate:+.1f}%", "Compared to Last Semester", "#1F4E78"),
            ("1ST-YEAR RETENTION", f"{retention_rate:.1f}%", "Freshmen Continuing Studies", "#2E7D32"),
            ("ON-TIME GRADUATION", f"{grad_rate:.1f}%", "4th-Year Graduating Rate", "#00695C"),
            ("SUBJECT PASSING RATE", f"{pass_rate:.1f}%", "Grades 1.00 to 3.00", "#800000"),
            ("BOARD EXAM PASSING", f"{board_rate:.1f}%" if board_rate > 0 else "Non-Board",
             f"National Avg: {nat_avg:.1f}%" if nat_avg > 0 else "IT & CS Courses", "#B26A00")
        ]

        for title, val, sub, color in cards:
            card = tk.Frame(kpi_frame, bg="white", bd=1, relief="solid", padx=12, pady=10)
            card.pack(side="left", fill="both", expand=True, padx=5)
            tk.Label(card, text=title, font=("Segoe UI", 9, "bold"), fg="#555555", bg="white").pack(anchor="w")
            tk.Label(card, text=val, font=("Segoe UI", 18, "bold"), fg=color, bg="white").pack(anchor="w", pady=(3, 1))
            tk.Label(card, text=sub, font=("Segoe UI", 8), fg="#777777", bg="white").pack(anchor="w")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=100)
        fig.patch.set_facecolor("#F4F6F9")
        self._active_figs.append(fig)
        hover_targets = []

        if len(dff) > 0:
            t_grp = dff.groupby("Academic_Term").agg(
                Students=("Student_ID", "count"),
                Stayed=("Dropped_Out", lambda x: (1 - x.mean()) * 100)
            ).sort_index()
            bars1 = ax1.bar(t_grp.index, t_grp["Students"], color="#800000", alpha=0.85, label="Enrolled Students")
            ax1.set_title("Student Enrollment & Retention Over Time (Hover bars/dots)", fontsize=10, fontweight="bold")
            ax1.set_ylabel("Number of Students", color="#800000")
            ax1.tick_params(axis="x", rotation=18)

            ax1_twin = ax1.twinx()
            line1, = ax1_twin.plot(t_grp.index, t_grp["Stayed"], color="#2E7D32", marker="o", markersize=8,
                                   linewidth=2.5, label="Retention Rate")
            ax1_twin.set_ylabel("Students Who Stayed (%)", color="#2E7D32")
            ax1_twin.set_ylim(50, 102)

            hover_targets.append({"type": "line", "ax": ax1_twin, "line": line1, "labels": t_grp.index.tolist(),
                                  "fmt": "Semester: {label}\nRetention Rate: {val:.1f}%"})
            hover_targets.append({"type": "bar", "ax": ax1, "bars": bars1, "labels": t_grp.index.tolist(),
                                  "fmt": "Semester: {label}\nEnrolled: {val:.0f} Students"})

        b_df = ops[ops["Board_Pass_Rate"] > 0]
        if len(b_df) > 0:
            x = np.arange(len(b_df["College"]))
            w = 0.35
            bars_bat = ax2.bar(x - w / 2, b_df["Board_Pass_Rate"], width=w, color="#1F4E78",
                               label="BatStateU ARASOF-Nasugbu")
            bars_nat = ax2.bar(x + w / 2, b_df["National_Board_Avg"], width=w, color="#B0BEC5",
                               label="National Average")
            ax2.set_xticks(x)
            ax2.set_xticklabels(b_df["College"])
            ax2.set_ylim(0, 108)
            ax2.set_ylabel("Passing Rate (%)")
            ax2.set_title("Board Exam Passing Rate vs. National Average", fontsize=10, fontweight="bold")
            ax2.legend()

            c_labels = b_df["College"].tolist()
            hover_targets.append({"type": "bar", "ax": ax2, "bars": bars_bat, "labels": c_labels,
                                  "fmt": "{label} (ARASOF-Nasugbu)\nBoard Passing: {val:.1f}%"})
            hover_targets.append({"type": "bar", "ax": ax2, "bars": bars_nat, "labels": c_labels,
                                  "fmt": "{label} (National Avg)\nBoard Passing: {val:.1f}%"})
        else:
            ax2.text(0.5, 0.5, "CICS offers Non-Board Courses (BSIT & BSCS)", ha="center", va="center", fontsize=11)
            ax2.set_title("Board Exam Performance", fontsize=10, fontweight="bold")

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tab_exec)
        canvas.draw()
        enable_chart_hover(fig, canvas, hover_targets)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    # --------------------------------------------------------------------------
    # TAB 2: BATANGAS MUNICIPALITIES, FACULTY & CAMPUS BUDGET
    # --------------------------------------------------------------------------
    def _render_tab2_operations(self, dff, ops):
        for widget in self.tab_ops.winfo_children():
            widget.destroy()

        scaled_students = len(dff) * 5.5
        total_faculty = ops["Active_Faculty"].sum()
        student_faculty_ratio = scaled_students / total_faculty if total_faculty > 0 else 0
        faculty_load = ops["Faculty_Load_Compliance"].mean()
        scholarship_rate = ops["Scholarship_Release_Rate"].mean()
        budget_used = ops["Budget_Used_Rate"].mean()
        old_hours = ops["Old_Report_Hours"].mean()
        new_hours = ops["New_BI_Report_Hours"].mean()

        kpi_frame = tk.Frame(self.tab_ops, bg="#F4F6F9")
        kpi_frame.pack(fill="x", pady=6)

        ops_cards = [
            ("STUDENTS PER TEACHER", f"{student_faculty_ratio:.0f} : 1", f"Total Teachers: {total_faculty}", "#1F4E78"),
            ("PROPER TEACHER LOAD", f"{faculty_load:.1f}%", "Following Standard Units", "#2E7D32"),
            ("SCHOLARSHIP RELEASE", f"{scholarship_rate:.1f}%", "Free Tuition / UniFAST", "#00695C"),
            ("CAMPUS BUDGET USED", f"{budget_used:.1f}%", "Allocated Funds Utilized", "#800000"),
            ("REPORT PREPARATION", f"{new_hours * 60:.0f} mins", f"Faster than {old_hours:.0f} hrs manual", "#6A1B9A")
        ]

        for title, val, sub, color in ops_cards:
            card = tk.Frame(kpi_frame, bg="white", bd=1, relief="solid", padx=12, pady=10)
            card.pack(side="left", fill="both", expand=True, padx=5)
            tk.Label(card, text=title, font=("Segoe UI", 9, "bold"), fg="#555555", bg="white").pack(anchor="w")
            tk.Label(card, text=val, font=("Segoe UI", 18, "bold"), fg=color, bg="white").pack(anchor="w", pady=(3, 1))
            tk.Label(card, text=sub, font=("Segoe UI", 8), fg="#777777", bg="white").pack(anchor="w")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=100)
        fig.patch.set_facecolor("#F4F6F9")
        self._active_figs.append(fig)
        hover_targets = []

        if len(dff) > 0:
            muni_counts = dff["Municipality"].value_counts().sort_values()
            bars_muni = ax1.barh(muni_counts.index, muni_counts.values, color="#1F4E78")
            ax1.set_title("Where ARASOF Students Live (Batangas District 1)", fontsize=10, fontweight="bold")
            ax1.set_xlabel("Number of Students")
            hover_targets.append(
                {"type": "bar", "horizontal": True, "ax": ax1, "bars": bars_muni, "labels": muni_counts.index.tolist(),
                 "fmt": "Municipality: {label}\nStudents: {val:.0f}"})

        x = np.arange(len(ops["College"]))
        w = 0.25
        b_load = ax2.bar(x - w, ops["Faculty_Load_Compliance"], width=w, color="#1F4E78", label="Teacher Load %")
        b_scho = ax2.bar(x, ops["Scholarship_Release_Rate"], width=w, color="#2E7D32", label="Scholarship Release %")
        b_budg = ax2.bar(x + w, ops["Budget_Used_Rate"], width=w, color="#800000", label="Budget Used %")
        ax2.set_xticks(x)
        ax2.set_xticklabels(ops["College"])
        ax2.set_ylim(75, 104)
        ax2.set_ylabel("Percentage (%)")
        ax2.set_title("Teacher Load, Scholarship & Budget Efficiency by College", fontsize=10, fontweight="bold")
        ax2.legend(loc="lower right", fontsize=8)

        col_labels = ops["College"].tolist()
        hover_targets.append({"type": "bar", "ax": ax2, "bars": b_load, "labels": col_labels,
                              "fmt": "{label} – Proper Teacher Load\nCompliance: {val:.1f}%"})
        hover_targets.append({"type": "bar", "ax": ax2, "bars": b_scho, "labels": col_labels,
                              "fmt": "{label} – Scholarship Release\nRate: {val:.1f}%"})
        hover_targets.append({"type": "bar", "ax": ax2, "bars": b_budg, "labels": col_labels,
                              "fmt": "{label} – Campus Budget\nUsed: {val:.1f}%"})

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tab_ops)
        canvas.draw()
        enable_chart_hover(fig, canvas, hover_targets)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    # --------------------------------------------------------------------------
    # TAB 3: STUDENT DROPOUT PREDICTOR (EARLY-WARNING SYSTEM)
    # --------------------------------------------------------------------------
    def _render_tab3_prediction(self, dff):
        for widget in self.tab_pred.winfo_children():
            widget.destroy()

        top_pane = tk.Frame(self.tab_pred, bg="#F4F6F9")
        top_pane.pack(fill="both", expand=True, padx=6, pady=6)

        left_frame = tk.Frame(top_pane, bg="white", bd=1, relief="solid")
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

        fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=100)
        self._active_figs.append(fig)
        importances = pd.Series(self.predictor.model.feature_importances_ * 100,
                                index=self.predictor.friendly_names).sort_values()
        bars_imp = ax.barh(importances.index, importances.values, color="#00695C")
        ax.set_title("Top Factors Affecting Student Dropout (Hover bars)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Impact Weight (%)")
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=left_frame)
        canvas.draw()
        enable_chart_hover(fig, canvas, [
            {"type": "bar", "horizontal": True, "ax": ax, "bars": bars_imp, "labels": importances.index.tolist(),
             "fmt": "Factor: {label}\nDropout Impact: {val:.1f}%"}
        ])
        canvas.get_tk_widget().pack(fill="both", expand=True)

        sim_frame = tk.LabelFrame(top_pane, text=" Test a Student Profile (Dropout Risk Checker) ",
                                  font=("Segoe UI", 10, "bold"), bg="white", padx=15, pady=8)
        sim_frame.pack(side="right", fill="both", expand=True)

        self.sim_midterm = tk.DoubleVar(value=3.00)
        self.sim_attend = tk.DoubleVar(value=76.0)
        self.sim_logins = tk.IntVar(value=4)
        self.sim_travel = tk.IntVar(value=55)
        self.sim_year = tk.IntVar(value=1)
        self.sim_scholar = tk.IntVar(value=1)

        controls = [
            ("Midterm Grade (1.00 Best - 5.00 Fail):", self.sim_midterm, 1.0, 5.0, 0.25),
            ("Class Attendance (%):", self.sim_attend, 50.0, 100.0, 1.0),
            ("Weekly Portal / Online Logins:", self.sim_logins, 0, 25, 1),
            ("Travel Time to Campus (Mins):", self.sim_travel, 10, 90, 5),
            ("Year Level (1st to 4th Year):", self.sim_year, 1, 4, 1),
            ("Has Scholarship (1 = Yes, 0 = No):", self.sim_scholar, 0, 1, 1)
        ]

        for idx, (lbl, var, frm, to, res) in enumerate(controls):
            tk.Label(sim_frame, text=lbl, font=("Segoe UI", 9), bg="white").grid(row=idx, column=0, sticky="w", pady=1)
            tk.Scale(sim_frame, variable=var, from_=frm, to=to, resolution=res, orient="horizontal", length=180,
                     bg="white").grid(row=idx, column=1, padx=8)

        self.pred_result_lbl = tk.Label(sim_frame, text="Move sliders and click 'Check Student Risk'",
                                        font=("Segoe UI", 10, "bold"), fg="#1F4E78", bg="white")
        self.pred_result_lbl.grid(row=6, column=0, columnspan=2, pady=(6, 4))

        tk.Button(sim_frame, text="Check Student Risk", command=self.run_prediction, bg="#800000", fg="white",
                  font=("Segoe UI", 10, "bold"), padx=14, pady=3).grid(row=7, column=0, columnspan=2)

        table_frame = tk.LabelFrame(self.tab_pred,
                                    text=" Students Needing Immediate Guidance Counseling & Academic Support ",
                                    font=("Segoe UI", 10, "bold"), bg="white")
        table_frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        cols = ("Student_ID", "College", "Program", "Municipality", "Travel_Mins", "Midterm_Grade", "Attendance_Rate",
                "Dropout_Risk_Percent")
        headers = {
            "Student_ID": "Student ID",
            "College": "College",
            "Program": "Course / Program",
            "Municipality": "Batangas Town",
            "Travel_Mins": "Travel Time",
            "Midterm_Grade": "Midterm Grade",
            "Attendance_Rate": "Attendance",
            "Dropout_Risk_Percent": "Chance of Dropping Out"
        }

        tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=6)
        for c in cols:
            tree.heading(c, text=headers[c])
            tree.column(c, width=135, anchor="center")

        high_risk_df = dff.sort_values(by="Dropout_Risk_Percent", ascending=False).head(25)
        for _, row in high_risk_df.iterrows():
            tree.insert("", "end", values=(
                row["Student_ID"], row["College"], row["Program"], row["Municipality"],
                f"{row['Travel_Mins']} mins", f"{row['Midterm_Grade']:.2f}", f"{row['Attendance_Rate']:.1f}%",
                f"{row['Dropout_Risk_Percent']:.1f}%"
            ))
        tree.pack(fill="both", expand=True, padx=6, pady=6)

    def run_prediction(self):
        risk_pct = self.predictor.predict_student(
            self.sim_midterm.get(),
            self.sim_attend.get(),
            self.sim_logins.get(),
            self.sim_travel.get(),
            self.sim_year.get(),
            self.sim_scholar.get()
        )
        if risk_pct >= 65.0:
            status, color = "HIGH RISK — Needs Immediate Counseling & Tutoring", "#C62828"
        elif risk_pct >= 35.0:
            status, color = "MODERATE RISK — Adviser Should Check In", "#EF6C00"
        else:
            status, color = "LOW RISK — Student is Doing Well", "#2E7D32"
        self.pred_result_lbl.config(text=f"Chance of Dropping Out: {risk_pct:.1f}% ({status})", fg=color)


if __name__ == "__main__":
    df_students, df_college_ops = generate_arasof_dataset(n_records=1000)
    predictor = DropoutPredictor(df_students)
    app = BatStateUARASOFDashboard(df_students, df_college_ops, predictor)
    app.mainloop()
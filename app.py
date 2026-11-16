import pyodbc
from flask import Flask, redirect, render_template, request, session, url_for

app = Flask(__name__)
# Секретний ключ необхідний для безпечної роботи сесій (щоб сайт "пам'ятав" логін)
app.secret_key = "hired_employee_secret_key_2026"


# Функція підключення до бази даних (перевірте ваші налаштування SERVER)
def get_db_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost\\SQLEXPRESS;"  # Замініть на свій сервер, якщо він відрізняється
        "DATABASE=HiredEmployeeAlertDB;"
        "Trusted_Connection=yes;"
    )
    return conn


# Головна сторінка автоматично перенаправляє на логін
@app.route("/")
def index():
    return redirect(url_for("login"))


# Авторизація
@app.route("/login", methods=["POST", "GET"])
def login():
    # Якщо користувач вже в системі — одразу на панель
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT EmployeeID, FullName, PasswordHash, RoleId FROM Employees WHERE Email = ?",
                (email,),
            )
            user = cursor.fetchone()
            conn.close()

            if user and user.PasswordHash == password:
                # Зберігаємо дані користувача у сесії
                session["user_id"] = user.EmployeeID
                session["user_name"] = user.FullName
                session["role_id"] = user.RoleId

                return redirect(url_for("dashboard"))
            else:
                return render_template("index.html", error="Невірний email або пароль!")
        except Exception as e:
            return f"Помилка бази даних: {str(e)}"

    return render_template("index.html")


# Головний розподільник (роутер) після входу
@app.route("/dashboard")
def dashboard():
    # Якщо сесії немає (не залогінений) — на сторінку входу
    if "user_id" not in session:
        return redirect(url_for("login"))

    role_id = session["role_id"]

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        if role_id == 1:  # HR
            return render_template("hr_dashboard.html", user_name=session["user_name"])

        elif role_id == 2:  # IT
            # Завдання налаштування (Onboarding) - Звільнені приховані
            cursor.execute("""
                SELECT t.TaskID, h.FirstName + ' ' + h.LastName AS EmployeeName,
                       t.TaskDescription, t.Status
                FROM OnboardingTasks t
                JOIN HiredEmployees h ON t.HiredEmployeeID = h.HiredEmployeeID
                WHERE t.Status != 'Completed' AND h.Status != 'Fired'
            """)
            tasks = cursor.fetchall()

            # Список речей, які треба забрати у звільнених (Offboarding)
            cursor.execute("""
                SELECT eq.EquipmentID, h.FirstName + ' ' + h.LastName AS EmployeeName, eq.ItemName
                FROM IssuedEquipment eq
                JOIN HiredEmployees h ON eq.HiredEmployeeID = h.HiredEmployeeID
                WHERE h.Status = 'Fired' AND eq.ReturnStatus = 'Issued'
            """)
            return_items = cursor.fetchall()

            return render_template(
                "it_dashboard.html",
                user_name=session["user_name"],
                tasks=tasks,
                return_items=return_items,
            )

        elif role_id == 4:  # Manager
            cursor.execute("""
                SELECT a.AlertID, h.FirstName + ' ' + h.LastName AS EmployeeName, a.MessageText,
                       FORMAT(a.SentAt, 'dd.MM.yyyy HH:mm') AS CreatedAt
                FROM Alerts a
                JOIN HiredEmployees h ON a.HiredEmployeeID = h.HiredEmployeeID
                ORDER BY a.AlertID DESC
            """)
            alerts = cursor.fetchall()
            return render_template(
                "manager_dashboard.html", user_name=session["user_name"], alerts=alerts
            )

        elif role_id == 3:  # Звичайний працівник
            # Шукаємо обладнання, де ім'я та прізвище збігаються з іменем користувача
            cursor.execute(
                """
                SELECT eq.ItemName, FORMAT(eq.IssueDate, 'dd.MM.yyyy') AS IssueDate
                FROM IssuedEquipment eq
                JOIN HiredEmployees h ON eq.HiredEmployeeID = h.HiredEmployeeID
                WHERE (h.FirstName + ' ' + h.LastName) = ?
            """,
                (session["user_name"],),
            )
            my_equipment = cursor.fetchall()
            return render_template(
                "employee_dashboard.html", user_name=session["user_name"], equipment=my_equipment
            )

        else:
            user_name = session["user_name"]
            return f"<h1>Вітаємо, {user_name}!</h1><p>Панель для вашої ролі ще розробляється.</p>"

    finally:
        conn.close()


# Вихід із системи
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# Додавання працівника (Доступно тільки HR)
@app.route("/add_employee", methods=["POST"])
def add_employee():
    # Перевірка доступу (тільки HR - role_id 1)
    if "user_id" not in session or session["role_id"] != 1:
        return redirect(url_for("login"))

    # Отримання даних з форми
    first_name = request.form.get("first_name")
    last_name = request.form.get("last_name")
    position = request.form.get("position")
    department = request.form.get("department")  # НОВЕ: отримуємо відділ з форми
    hire_date = request.form.get("hire_date")

    it_task_type = request.form.get("it_task_type")
    custom_task = request.form.get("custom_task")

    target_department_id = 2  # Відправляємо сповіщення IT-відділу

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # НОВЕ: Додаємо Department у базу даних при реєстрації
        cursor.execute(
            """
            INSERT INTO HiredEmployees (FirstName, LastName, Position, Department, HireDate, Status)
            OUTPUT INSERTED.HiredEmployeeID
            VALUES (?, ?, ?, ?, ?, 'Active')
        """,
            (first_name, last_name, position, department, hire_date),
        )

        # Отримуємо ID щойно створеного працівника
        new_employee_id = cursor.fetchone()[0]

        # Формуємо текст завдання для IT
        task_desc = custom_task if it_task_type == "custom" else it_task_type

        # Створюємо завдання для IT
        cursor.execute(
            """
            INSERT INTO OnboardingTasks (HiredEmployeeID, TaskDescription, Status)
            VALUES (?, ?, 'Pending')
        """,
            (new_employee_id, task_desc),
        )

        # Створюємо сповіщення для Керівника/IT
        cursor.execute(
            """
            INSERT INTO Alerts (HiredEmployeeID, TargetDepartmentID, AlertMessage, IsRead)
            VALUES (?, ?, 'New employee added, onboarding required', 0)
        """,
            (new_employee_id, target_department_id),
        )

        conn.commit()
        conn.close()

        # Повертаємо HR на головну панель після успішного додавання
        return redirect(url_for("dashboard"))

    except Exception as e:
        return f"Помилка бази даних: {str(e)}"


# Виконання завдання IT-відділом
@app.route("/complete_task/<int:task_id>", methods=["POST"])
def complete_task(task_id):
    if "user_id" not in session or session["role_id"] != 2:
        return redirect(url_for("login"))

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 1. Оновлюємо статус
        cursor.execute(
            "UPDATE OnboardingTasks SET Status = 'Completed' WHERE TaskID = ?", (task_id,)
        )

        # 2. Шукаємо працівника
        cursor.execute("SELECT HiredEmployeeID FROM OnboardingTasks WHERE TaskID = ?", (task_id,))
        hired_employee_id = cursor.fetchone()[0]

        # 3. Створюємо Alert
        manager_employee_id = 3
        alert_message = (
            f"Завдання з налаштування (Task ID: {task_id}) виконано. Працівник готовий до роботи!"
        )

        cursor.execute(
            """
            INSERT INTO Alerts (HiredEmployeeID, RecipientID, MessageText, SentAt)
            VALUES (?, ?, ?, GETDATE())
        """,
            (hired_employee_id, manager_employee_id, alert_message),
        )

        conn.commit()

        # 4. Одразу витягуємо оновлений список завдань
        cursor.execute("""
            SELECT t.TaskID, h.FirstName + ' ' + h.LastName AS EmployeeName,
                   t.TaskDescription, t.Status
            FROM OnboardingTasks t
            JOIN HiredEmployees h ON t.HiredEmployeeID = h.HiredEmployeeID
            WHERE t.Status != 'Completed'
        """)
        tasks = cursor.fetchall()
        conn.close()

        return render_template("it_dashboard.html", user_name=session["user_name"], tasks=tasks)

    except Exception as e:
        return f"Помилка: {str(e)}"


# Сторінка "Структура компанії" (Список усіх працівників)
@app.route("/employees")
def employees():
    if "user_id" not in session:
        return redirect(url_for("login"))

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Отримуємо всіх працівників
        cursor.execute("""
            SELECT HiredEmployeeID, FirstName, LastName, Position, Department,
                   FORMAT(HireDate, 'dd.MM.yyyy') as HireDate, Status
            FROM HiredEmployees
            ORDER BY Department ASC, HiredEmployeeID ASC
        """)
        all_employees = cursor.fetchall()
        conn.close()

        # НОВЕ: Групуємо працівників по відділах
        active_grouped = {}
        fired_employees = []

        for emp in all_employees:
            if emp.Status == "Fired":
                fired_employees.append(emp)
            else:
                # Якщо відділ порожній (NULL), відносимо їх до "Нерозподілених"
                dept = emp.Department if emp.Department else "Нерозподілені (Без відділу)"

                if dept not in active_grouped:
                    active_grouped[dept] = []
                active_grouped[dept].append(emp)

        # Передаємо згруповані словники у шаблон
        return render_template(
            "employees.html", active_grouped=active_grouped, fired_employees=fired_employees
        )
    except Exception as e:
        return f"Помилка бази даних: {str(e)}"


# Сторінка "Облік обладнання"
@app.route("/equipment", methods=["GET", "POST"])
def equipment():
    # Доступ мають лише HR (1) та Керівник (4)
    if "user_id" not in session or session["role_id"] not in [1, 4]:
        return redirect(url_for("login"))

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Якщо відправлено форму (видача нового обладнання)
        if request.method == "POST":
            emp_id = request.form.get("employee_id")
            item_name = request.form.get("item_name")

            cursor.execute(
                """
                INSERT INTO IssuedEquipment (HiredEmployeeID, ItemName, IssueDate)
                VALUES (?, ?, GETDATE())
            """,
                (emp_id, item_name),
            )
            conn.commit()
            return redirect(url_for("equipment"))  # Оновлюємо сторінку

        # Якщо просто завантажуємо сторінку:
        # 1. Отримуємо історію виданого обладнання
        cursor.execute("""
            SELECT eq.EquipmentID, h.FirstName + ' ' + h.LastName AS EmployeeName, eq.ItemName,
                   FORMAT(eq.IssueDate, 'dd.MM.yyyy HH:mm') AS IssueDate
            FROM IssuedEquipment eq
            JOIN HiredEmployees h ON eq.HiredEmployeeID = h.HiredEmployeeID
            WHERE h.Status != 'Fired'
            ORDER BY eq.EquipmentID DESC
        """)
        equipment_list = cursor.fetchall()

        # 2. Отримуємо список працівників для випадаючого списку у формі
        cursor.execute("""
            SELECT HiredEmployeeID, FirstName + ' ' + LastName AS FullName
            FROM HiredEmployees
            ORDER BY HiredEmployeeID DESC
        """)
        employees = cursor.fetchall()

        conn.close()

        return render_template("equipment.html", equipment=equipment_list, employees=employees)

    except Exception as e:
        return f"Помилка бази даних: {str(e)}"


# Редагування працівника (Тільки для HR)
@app.route("/edit_employee/<int:emp_id>", methods=["GET", "POST"])
def edit_employee(emp_id):
    if "user_id" not in session or session["role_id"] != 1:
        return redirect(url_for("login"))

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        if request.method == "POST":
            first_name = request.form.get("first_name")
            last_name = request.form.get("last_name")
            position = request.form.get("position")
            department = request.form.get("department")  # Отримуємо відділ з форми

            cursor.execute(
                """
                UPDATE HiredEmployees
                SET FirstName = ?, LastName = ?, Position = ?, Department = ?
                WHERE HiredEmployeeID = ?
            """,
                (first_name, last_name, position, department, emp_id),
            )
            conn.commit()
            conn.close()
            return redirect(url_for("employees"))

        # Якщо це GET-запит, завантажуємо поточні дані працівника (додано Department)
        cursor.execute(
            """
            SELECT HiredEmployeeID, FirstName, LastName, Position, Department
            FROM HiredEmployees
            WHERE HiredEmployeeID = ?
            """,
            (emp_id,),
        )
        emp = cursor.fetchone()
        conn.close()

        return render_template("edit_employee.html", emp=emp)
    except Exception as e:
        return f"Помилка: {str(e)}"


# Звільнення працівника (тільки для HR)
@app.route("/fire_employee/<int:emp_id>", methods=["POST"])
def fire_employee(emp_id):
    if "user_id" not in session or session["role_id"] != 1:
        return redirect(url_for("login"))

    conn = get_db_connection()
    cursor = conn.cursor()
    # Змінюємо статус працівника на "Звільнений"
    cursor.execute(
        "UPDATE HiredEmployees SET Status = 'Fired' WHERE HiredEmployeeID = ?", (emp_id,)
    )
    conn.commit()
    conn.close()
    return redirect(url_for("employees"))


# Повернення одиниці обладнання (тільки для IT)
@app.route("/return_equipment/<int:eq_id>", methods=["POST"])
def return_equipment(eq_id):
    if "user_id" not in session or session["role_id"] != 2:
        return redirect(url_for("login"))

    conn = get_db_connection()
    cursor = conn.cursor()
    # Відмічаємо конкретну річ як "Повернуту"
    cursor.execute(
        "UPDATE IssuedEquipment SET ReturnStatus = 'Returned' WHERE EquipmentID = ?", (eq_id,)
    )
    conn.commit()
    conn.close()
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    app.run(debug=True)

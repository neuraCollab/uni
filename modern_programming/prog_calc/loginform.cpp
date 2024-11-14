#include "loginform.h"
#include "ui_loginform.h"

LoginForm::LoginForm(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::LoginForm)
{
    setWindowTitle("Login Form");
    setGeometry(100, 100, 300, 200);

    // Создаем текстовое поле для имени пользователя
    QLabel *usernameLabel = new QLabel("Username:", this);
    usernameLineEdit = new QLineEdit(this);

    // Создаем текстовое поле для пароля
    QLabel *passwordLabel = new QLabel("Password:", this);
    passwordLineEdit = new QLineEdit(this);
    passwordLineEdit->setEchoMode(QLineEdit::Password);

    // Создаем метку для ошибки
    errorLabel = new QLabel(this);
    errorLabel->setStyleSheet("QLabel { color : red; }");

    // Создаем кнопку входа
    loginButton = new QPushButton("Login", this);
    connect(loginButton, &QPushButton::clicked, this, &LoginForm::on_pushButton_clicked);

    // Компонуем элементы интерфейса
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    QHBoxLayout *usernameLayout = new QHBoxLayout();
    QHBoxLayout *passwordLayout = new QHBoxLayout();

    usernameLayout->addWidget(usernameLabel);
    usernameLayout->addWidget(usernameLineEdit);

    passwordLayout->addWidget(passwordLabel);
    passwordLayout->addWidget(passwordLineEdit);

    mainLayout->addLayout(usernameLayout);
    mainLayout->addLayout(passwordLayout);
    mainLayout->addWidget(loginButton);
    mainLayout->addWidget(errorLabel);

    setLayout(mainLayout);
}

LoginForm::~LoginForm()
{
    delete ui;
}

void LoginForm::onCalculatorClosed()
{
    this->show();
}


void LoginForm::on_pushButton_clicked()
{

    QString username = usernameLineEdit->text();
    QString password = passwordLineEdit->text();

    if (username == "admin" && password == "1234") {
        calculatorForm = new CalculatorForm(this);

        qDebug() << "Open calc";

        connect(calculatorForm, &CalculatorForm::calculatorClosed, this, &LoginForm::onCalculatorClosed);

        calculatorForm->show();
        // this->hide(); // вобщем когда он скрывается, то сразу же скрывается форма калькулятора. Я не разобрался как ее нормально скрывать
    } else {
        ui->errorLabel->setText("Неправильный логин или пароль");
    }
}




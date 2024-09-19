#include "loginform.h"
#include "ui_loginform.h"

LoginForm::LoginForm(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::LoginForm)
{
    ui->setupUi(this);
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

    QString username = ui->usernameLineEdit->text();
    QString password = ui->passwordLineEdit->text();

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




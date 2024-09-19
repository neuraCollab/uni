#include "calculatorform.h"
#include "ui_calculatorform.h"

CalculatorForm::CalculatorForm(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::CalculatorForm)
{
    ui->setupUi(this);

    connect(ui->pushButton_0, &QPushButton::clicked, this, &CalculatorForm::digit_pressed);
    connect(ui->pushButton_1, &QPushButton::clicked, this, &CalculatorForm::digit_pressed);
    connect(ui->pushButton_2, &QPushButton::clicked, this, &CalculatorForm::digit_pressed);
    connect(ui->pushButton_3, &QPushButton::clicked, this, &CalculatorForm::digit_pressed);
    connect(ui->pushButton_4, &QPushButton::clicked, this, &CalculatorForm::digit_pressed);
    connect(ui->pushButton_5, &QPushButton::clicked, this, &CalculatorForm::digit_pressed);
    connect(ui->pushButton_6, &QPushButton::clicked, this, &CalculatorForm::digit_pressed);
    connect(ui->pushButton_7, &QPushButton::clicked, this, &CalculatorForm::digit_pressed);
    connect(ui->pushButton_8, &QPushButton::clicked, this, &CalculatorForm::digit_pressed);
    connect(ui->pushButton_9, &QPushButton::clicked, this, &CalculatorForm::digit_pressed);
}



CalculatorForm::~CalculatorForm()
{
    emit calculatorClosed();
    delete ui;
}

void CalculatorForm::digit_pressed()
{
    QPushButton *button = (QPushButton*)sender();
    QString buttonValue = button->text();
    QString currentValue = ui->display->text();

    if (currentValue == "0") {
        ui->display->setText(buttonValue);
    } else {
        // Иначе добавляем цифру к текущему значению
        ui->display->setText(currentValue + buttonValue);
    }
}

void CalculatorForm::on_addButton_clicked()
{
    firstNumber = ui->display->text().toDouble();
    pendingOperator = "+";
    ui->display->clear();
}

void CalculatorForm::on_subtractButton_clicked()
{
    firstNumber = ui->display->text().toDouble();
    pendingOperator = "-";
    ui->display->clear();
}


void CalculatorForm::on_multiplyButton_clicked()
{
    firstNumber = ui->display->text().toDouble();
    pendingOperator = "*";
    ui->display->clear();
}


void CalculatorForm::on_divideButton_clicked()
{
    firstNumber = ui->display->text().toDouble();
    pendingOperator = "/";
    ui->display->clear();
}

void CalculatorForm::on_equalsButton_clicked()
{
    secondNumber = ui->display->text().toDouble();

    double result = 0;

    if (pendingOperator == "+") {
        result = firstNumber + secondNumber;
    }
    else if (pendingOperator == "-") {
        result = firstNumber - secondNumber;
    }
    else if (pendingOperator == "*") {
        result = firstNumber * secondNumber;
    }
    else if (pendingOperator == "/") {
        if (secondNumber != 0) {
            result = firstNumber / secondNumber;
        } else {
            ui->display->setText("Ошибка деления на 0");
            return;
        }
    }

    ui->display->setText(QString::number(result));
}


void CalculatorForm::on_clearButton_clicked()
{
    firstNumber = 0;
    secondNumber = 0;
    pendingOperator = "";
    ui->display->clear();
}


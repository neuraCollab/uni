#include "calculatorform.h"
#include "ui_calculatorform.h"
#include <QPushButton>
#include <QWidget>
#include <QGridLayout>
#include <QFormLayout>
#include <QLineEdit>
#include <QSignalMapper>

CalculatorForm::CalculatorForm(QWidget *parent)
    : QDialog(parent)
{

    setWindowTitle("Form");
    setGeometry(0, 0, 400, 300);

    // Создаем главное поле для вывода результата

    display = new QLineEdit("0", this);
    display->setGeometry(0, 30, 391, 61);

    // Инициализация signalMapper
    signalMapper = new QSignalMapper(this);

    // Создаем кнопки для цифр и добавляем их в сетку
    QGridLayout *gridLayout = new QGridLayout();
    QPushButton *buttons[10];

    for (int i = 1; i < 10; ++i) {
        buttons[i] = new QPushButton(QString::number(i), this);
        connect(buttons[i], &QPushButton::clicked, this, &CalculatorForm::digit_pressed); // Подключаем сигнал к слоту
        gridLayout->addWidget(buttons[i], i / 3, i % 3); // Располагаем кнопки в сетке
    }
    buttons[0] = new QPushButton("0", this);
    connect(buttons[0], &QPushButton::clicked, this, &CalculatorForm::digit_pressed); // Подключаем сигнал к слоту
    gridLayout->addWidget(buttons[0], 3, 1);

    // Создаем контейнер для сетки с кнопками
    QWidget *gridLayoutWidget = new QWidget(this);
    gridLayoutWidget->setGeometry(-10, 100, 295, 161);
    gridLayoutWidget->setLayout(gridLayout);

    // Создаем кнопки операций и добавляем их в форму
    QFormLayout *formLayout = new QFormLayout();
    QPushButton *divideButton = new QPushButton("/", this);
    QPushButton *multiplyButton = new QPushButton("x", this);
    QPushButton *subtractButton = new QPushButton("-", this);
    QPushButton *addButton = new QPushButton("+", this);

    connect(divideButton, &QPushButton::clicked, this, &CalculatorForm::on_divideButton_clicked);
    connect(multiplyButton, &QPushButton::clicked, this, &CalculatorForm::on_multiplyButton_clicked);
    connect(subtractButton, &QPushButton::clicked, this, &CalculatorForm::on_subtractButton_clicked);
    connect(addButton, &QPushButton::clicked, this, &CalculatorForm::on_addButton_clicked);


    formLayout->addWidget(divideButton);
    formLayout->addWidget(multiplyButton);
    formLayout->addWidget(subtractButton);
    formLayout->addWidget(addButton);

    // Создаем контейнер для кнопок операций
    QWidget *formLayoutWidget = new QWidget(this);
    formLayoutWidget->setGeometry(290, 100, 102, 161);
    formLayoutWidget->setLayout(formLayout);

    // Создаем кнопку равно
    QPushButton *equalsButton = new QPushButton("=", this);
    equalsButton->setGeometry(290, 270, 93, 29);
    connect(equalsButton, &QPushButton::clicked, this, &CalculatorForm::on_equalsButton_clicked);


    // Создаем кнопку очистки
    QPushButton *clearButton = new QPushButton("Отчистить", this);
    clearButton->setGeometry(0, 270, 93, 29);
    connect(clearButton, &QPushButton::clicked, this, &CalculatorForm::on_clearButton_clicked);

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
    QString currentValue = display->text(); // Используем display вместо ui->display

    if (currentValue == "0") {
        display->setText(buttonValue); // Используем display
    } else {
        display->setText(currentValue + buttonValue); // Используем display
    }
}

void CalculatorForm::on_addButton_clicked()
{
    firstNumber = display->text().toDouble(); // Замените ui->display на display
    pendingOperator = "+";
    display->clear(); // Замените ui->display на display
}

void CalculatorForm::on_subtractButton_clicked()
{
    firstNumber = display->text().toDouble(); // Замените ui->display на display
    pendingOperator = "-";
    display->clear(); // Замените ui->display на display
}

void CalculatorForm::on_multiplyButton_clicked()
{
    firstNumber = display->text().toDouble(); // Замените ui->display на display
    pendingOperator = "*";
    display->clear(); // Замените ui->display на display
}

void CalculatorForm::on_divideButton_clicked()
{
    firstNumber = display->text().toDouble(); // Замените ui->display на display
    pendingOperator = "/";
    display->clear(); // Замените ui->display на display
}

void CalculatorForm::on_equalsButton_clicked()
{
    secondNumber = display->text().toDouble(); // Замените ui->display на display
    qDebug() << "pendingOperator: " << pendingOperator;
    qDebug() << "firstNumber: " << firstNumber;
    qDebug() << "secondNumber: " << secondNumber;

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
            display->setText("Ошибка деления на 0"); // Замените ui->display на display
            return;
        }
    }

    qDebug() << "result: " << result;

    display->setText(QString::number(result)); // Замените ui->display на display
}

void CalculatorForm::on_clearButton_clicked()
{
    firstNumber = 0;
    secondNumber = 0;
    pendingOperator = "";
    display->clear(); // Замените ui->display на display
}

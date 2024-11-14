#ifndef CALCULATORFORM_H
#define CALCULATORFORM_H

#include <QDialog>
#include <QLineEdit>
#include <QSignalMapper>

namespace Ui {
class CalculatorForm;
}

class CalculatorForm : public QDialog
{
    Q_OBJECT
signals:
    void calculatorClosed();

public:
    explicit CalculatorForm(QWidget *parent = nullptr);
    ~CalculatorForm();

private slots:
    void on_addButton_clicked();
    void on_subtractButton_clicked();
    void on_multiplyButton_clicked();
    void on_divideButton_clicked();
    void on_equalsButton_clicked();
    void on_clearButton_clicked();
    void digit_pressed();

private:
    double firstNumber;
    double secondNumber;
    QString pendingOperator;
    QLineEdit *display; // Объявляем переменную display здесь
    QSignalMapper *signalMapper;
    Ui::CalculatorForm *ui;

};



#endif // CALCULATORFORM_H

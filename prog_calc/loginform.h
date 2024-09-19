#ifndef LOGINFORM_H
#define LOGINFORM_H

#include <QDialog>
#include "calculatorform.h"

namespace Ui {
class LoginForm;
}

class LoginForm : public QDialog
{
    Q_OBJECT

public:
    explicit LoginForm(QWidget *parent = nullptr);
    ~LoginForm();

private slots:
    void on_pushButton_clicked();
    void onCalculatorClosed();

private:
    Ui::LoginForm *ui;
    CalculatorForm *calculatorForm;
};

#endif // LOGINFORM_H

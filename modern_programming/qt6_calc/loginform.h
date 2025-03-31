#ifndef LOGINFORM_H
#define LOGINFORM_H

#include <QDialog>
#include "calculatorform.h"
#include <QDialog>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>

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

    QLineEdit *usernameLineEdit;
    QLineEdit *passwordLineEdit;
    QLabel *errorLabel;
    QPushButton *loginButton;
};

#endif // LOGINFORM_H

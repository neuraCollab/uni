/********************************************************************************
** Form generated from reading UI file 'calculatorform.ui'
**
** Created by: Qt User Interface Compiler version 6.7.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CALCULATORFORM_H
#define UI_CALCULATORFORM_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_CalculatorForm
{
public:
    QWidget *gridLayoutWidget;
    QGridLayout *gridLayout;
    QPushButton *pushButton_8;
    QPushButton *pushButton_2;
    QPushButton *pushButton_4;
    QPushButton *pushButton_1;
    QPushButton *pushButton_7;
    QPushButton *pushButton_5;
    QPushButton *pushButton_6;
    QPushButton *pushButton_9;
    QPushButton *pushButton_3;
    QWidget *formLayoutWidget;
    QFormLayout *formLayout;
    QPushButton *divideButton;
    QPushButton *addButton;
    QPushButton *multiplyButton;
    QPushButton *subtractButton;
    QPushButton *equalsButton;
    QLineEdit *display;
    QPushButton *clearButton;
    QPushButton *pushButton_0;

    void setupUi(QWidget *CalculatorForm)
    {
        if (CalculatorForm->objectName().isEmpty())
            CalculatorForm->setObjectName("CalculatorForm");
        CalculatorForm->resize(400, 300);
        gridLayoutWidget = new QWidget(CalculatorForm);
        gridLayoutWidget->setObjectName("gridLayoutWidget");
        gridLayoutWidget->setGeometry(QRect(-10, 100, 295, 161));
        gridLayout = new QGridLayout(gridLayoutWidget);
        gridLayout->setObjectName("gridLayout");
        gridLayout->setContentsMargins(0, 0, 0, 0);
        pushButton_8 = new QPushButton(gridLayoutWidget);
        pushButton_8->setObjectName("pushButton_8");

        gridLayout->addWidget(pushButton_8, 2, 1, 1, 1);

        pushButton_2 = new QPushButton(gridLayoutWidget);
        pushButton_2->setObjectName("pushButton_2");

        gridLayout->addWidget(pushButton_2, 0, 1, 1, 1);

        pushButton_4 = new QPushButton(gridLayoutWidget);
        pushButton_4->setObjectName("pushButton_4");

        gridLayout->addWidget(pushButton_4, 1, 0, 1, 1);

        pushButton_1 = new QPushButton(gridLayoutWidget);
        pushButton_1->setObjectName("pushButton_1");

        gridLayout->addWidget(pushButton_1, 0, 0, 1, 1);

        pushButton_7 = new QPushButton(gridLayoutWidget);
        pushButton_7->setObjectName("pushButton_7");

        gridLayout->addWidget(pushButton_7, 2, 0, 1, 1);

        pushButton_5 = new QPushButton(gridLayoutWidget);
        pushButton_5->setObjectName("pushButton_5");

        gridLayout->addWidget(pushButton_5, 1, 1, 1, 1);

        pushButton_6 = new QPushButton(gridLayoutWidget);
        pushButton_6->setObjectName("pushButton_6");

        gridLayout->addWidget(pushButton_6, 1, 2, 1, 1);

        pushButton_9 = new QPushButton(gridLayoutWidget);
        pushButton_9->setObjectName("pushButton_9");

        gridLayout->addWidget(pushButton_9, 2, 2, 1, 1);

        pushButton_3 = new QPushButton(gridLayoutWidget);
        pushButton_3->setObjectName("pushButton_3");

        gridLayout->addWidget(pushButton_3, 0, 2, 1, 1);

        formLayoutWidget = new QWidget(CalculatorForm);
        formLayoutWidget->setObjectName("formLayoutWidget");
        formLayoutWidget->setGeometry(QRect(290, 100, 102, 161));
        formLayout = new QFormLayout(formLayoutWidget);
        formLayout->setObjectName("formLayout");
        formLayout->setContentsMargins(0, 0, 0, 0);
        divideButton = new QPushButton(formLayoutWidget);
        divideButton->setObjectName("divideButton");

        formLayout->setWidget(0, QFormLayout::LabelRole, divideButton);

        addButton = new QPushButton(formLayoutWidget);
        addButton->setObjectName("addButton");

        formLayout->setWidget(3, QFormLayout::LabelRole, addButton);

        multiplyButton = new QPushButton(formLayoutWidget);
        multiplyButton->setObjectName("multiplyButton");

        formLayout->setWidget(1, QFormLayout::LabelRole, multiplyButton);

        subtractButton = new QPushButton(formLayoutWidget);
        subtractButton->setObjectName("subtractButton");

        formLayout->setWidget(2, QFormLayout::LabelRole, subtractButton);

        equalsButton = new QPushButton(CalculatorForm);
        equalsButton->setObjectName("equalsButton");
        equalsButton->setGeometry(QRect(290, 270, 93, 29));
        display = new QLineEdit(CalculatorForm);
        display->setObjectName("display");
        display->setGeometry(QRect(0, 30, 391, 61));
        clearButton = new QPushButton(CalculatorForm);
        clearButton->setObjectName("clearButton");
        clearButton->setGeometry(QRect(0, 270, 93, 29));
        pushButton_0 = new QPushButton(CalculatorForm);
        pushButton_0->setObjectName("pushButton_0");
        pushButton_0->setGeometry(QRect(110, 270, 51, 31));

        retranslateUi(CalculatorForm);

        QMetaObject::connectSlotsByName(CalculatorForm);
    } // setupUi

    void retranslateUi(QWidget *CalculatorForm)
    {
        CalculatorForm->setWindowTitle(QCoreApplication::translate("CalculatorForm", "Form", nullptr));
        pushButton_8->setText(QCoreApplication::translate("CalculatorForm", "8", nullptr));
        pushButton_2->setText(QCoreApplication::translate("CalculatorForm", "2", nullptr));
        pushButton_4->setText(QCoreApplication::translate("CalculatorForm", "4", nullptr));
        pushButton_1->setText(QCoreApplication::translate("CalculatorForm", "1", nullptr));
        pushButton_7->setText(QCoreApplication::translate("CalculatorForm", "7", nullptr));
        pushButton_5->setText(QCoreApplication::translate("CalculatorForm", "5", nullptr));
        pushButton_6->setText(QCoreApplication::translate("CalculatorForm", "6", nullptr));
        pushButton_9->setText(QCoreApplication::translate("CalculatorForm", "9", nullptr));
        pushButton_3->setText(QCoreApplication::translate("CalculatorForm", "3", nullptr));
        divideButton->setText(QCoreApplication::translate("CalculatorForm", "/", nullptr));
        addButton->setText(QCoreApplication::translate("CalculatorForm", "+", nullptr));
        multiplyButton->setText(QCoreApplication::translate("CalculatorForm", "x", nullptr));
        subtractButton->setText(QCoreApplication::translate("CalculatorForm", "-", nullptr));
        equalsButton->setText(QCoreApplication::translate("CalculatorForm", "=", nullptr));
        display->setText(QCoreApplication::translate("CalculatorForm", "0", nullptr));
        clearButton->setText(QCoreApplication::translate("CalculatorForm", "\320\236\321\202\321\207\320\270\321\201\321\202\320\270\321\202\321\214", nullptr));
        pushButton_0->setText(QCoreApplication::translate("CalculatorForm", "0", nullptr));
    } // retranslateUi

};

namespace Ui {
    class CalculatorForm: public Ui_CalculatorForm {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CALCULATORFORM_H

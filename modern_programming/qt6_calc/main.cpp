#include "mainwindow.h"

#include <QApplication>
#include <QLocale>
#include <QTranslator>

#include "loginform.h"
#include "calculatorform.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    LoginForm login;

    QTranslator translator;
    const QStringList uiLanguages = QLocale::system().uiLanguages();
    for (const QString &locale : uiLanguages) {
        const QString baseName = "prog_calc_" + QLocale(locale).name();
        if (translator.load(":/i18n/" + baseName)) {
            a.installTranslator(&translator);
            break;
        }
    }

    if (login.exec() == QDialog::Accepted) {
        CalculatorForm calculator;
        calculator.show();

        return a.exec();
    } else {
        return 0;
    }
}

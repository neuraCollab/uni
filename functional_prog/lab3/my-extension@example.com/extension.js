import St from 'gi://St';
import { Extension } from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import * as PanelMenu from 'resource:///org/gnome/shell/ui/panelMenu.js';
import GLib from 'gi://GLib';
import Clutter from 'gi://Clutter';
import  Gio from 'gi://Gio';

export default class AuditExtension extends Extension {
    constructor(metadata) {
        super(metadata);
        this.eventLog = [];
        this.dialog = null; // Модальное окно
        this.dbusConnection = Gio.bus_get_sync(Gio.BusType.SESSION, null); // Получаем соединение DBus
        this.systemService = new Gio.DBusProxy({
            g_connection: this.dbusConnection,
            g_name: 'org.gnome.SystemMonitor',
            g_object_path: '/org/gnome/SystemMonitor',
            g_interface_name: 'org.gnome.SystemMonitor',
        });
    }

    enable() {
        // Создание индикатора на панели
        this._indicator = new PanelMenu.Button(0.0, this.metadata.name, false);

        // Блок для отображения статистики
        this._iconBox = new St.BoxLayout({ vertical: false, style_class: 'status-box' });

        this._cpuLabel = this.createStatLabel('CPU', 'utilities-system-monitor-symbolic');
        this._ramLabel = this.createStatLabel('RAM', 'media-memory-symbolic');
        this._batteryLabel = this.createStatLabel('BAT', 'battery-good-symbolic');

        this._iconBox.add_child(this._cpuLabel.box);
        this._iconBox.add_child(this._ramLabel.box);
        this._iconBox.add_child(this._batteryLabel.box);

        this._indicator.add_child(this._iconBox);
        Main.panel.addToStatusArea(this.uuid, this._indicator);

        // Обработка кликов по индикатору
        this._indicator.connect('button-press-event', this.showAuditMenu.bind(this));

        // Запуск мониторинга
        this.startAudit();
    }

    disable() {
        this._indicator?.destroy();
        this._indicator = null;
        this.stopAudit();
        this.dialog?.destroy();
        this.dialog = null;
    }

    startAudit() {
        // Обновляем статистику каждые 5 секунд
        this._timeoutId = GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, 5, () => {
            this.updateStats();
            return true; // Повторяем вызов
        });
    }

    stopAudit() {
        if (this._timeoutId) {
            GLib.source_remove(this._timeoutId);
            this._timeoutId = null;
        }
    }

    createStatLabel(name, iconName) {
        const box = new St.BoxLayout({ vertical: false, style_class: 'stat-box' });
        const icon = new St.Icon({
            icon_name: iconName,
            style_class: 'system-status-icon',
        });
        const label = new St.Label({ text: `${name}: --` });

        box.add_child(icon);
        box.add_child(label);

        return { box, label };
    }

    updateStats() {
        // Получаем данные о CPU, RAM и батарее через DBus
        this.systemService.GetCpuUsageAsync((result, error) => {
            if (!error) {
                this._cpuLabel.label.text = `CPU: ${result.toFixed(1)}%`;
            }
        });

        this.systemService.GetRamUsageAsync((result, error) => {
            if (!error) {
                this._ramLabel.label.text = `RAM: ${result.toFixed(1)}%`;
            }
        });

        this.systemService.GetBatteryStatusAsync((result, error) => {
            if (!error) {
                this._batteryLabel.label.text = `BAT: ${result}`;
            }
        });
    }

    showAuditMenu() {
        // Если окно уже открыто, просто закрываем
        if (this.dialog) {
            this.dialog.destroy();
            this.dialog = null;
            return;
        }

        // Создаем модальное окно
        this.dialog = new St.Bin({
            style_class: 'dialog',
            reactive: true,
            x_align: Clutter.ActorAlign.END,
            y_align: Clutter.ActorAlign.START,
        });

        const allocation = this._indicator.get_allocation_box();
        this.dialog.set_position(
            allocation.x1, 
            allocation.y2 + 5 // Ровно под кнопкой
        );

        const content = new St.BoxLayout({ vertical: true });
        const header = new St.Label({ text: 'Detailed System Statistics', style_class: 'header' });
        content.add_child(header);

        this.eventLog.slice(-5).forEach((entry) => {
            content.add_child(
                new St.Label({
                    text: `[${entry.timestamp}] ${entry.type}: ${entry.message}`,
                })
            );
        });

        // Добавляем кнопку закрытия
        const closeButton = new St.Button({
            label: 'Close',
            style_class: 'close-button',
        });
        closeButton.connect('clicked', () => {
            this.dialog.destroy();
            this.dialog = null;
        });
        content.add_child(closeButton);

        this.dialog.set_child(content);
        Main.layoutManager.addTopChrome(this.dialog);
    }
}

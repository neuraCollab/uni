const sqlite3 = require("sqlite3").verbose();

// Создаем подключение к базе данных
const db = new sqlite3.Database("analytics.db");

// Создание таблицы "сайты"
db.run(`CREATE TABLE IF NOT EXISTS sites (
    id INTEGER PRIMARY KEY,
    url TEXT,
    title TEXT,
    faviconUrl TEXT,
    description TEXT
  )`);

db.run(`CREATE TABLE IF NOT EXISTS visiters (
    id INTEGER PRIMARY KEY,
    token TEXT, 
    siteId INTEGER,
    FOREIGN KEY (siteId) REFERENCES sites(id)
  )`);
  
// Создание таблицы "события"
db.run(`CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    visiterId INTEGER,
    eventType TEXT,
    eventValue TEXT,
    timestamp TEXT,
    FOREIGN KEY (visiterId) REFERENCES visiters(id)
  )`);

module.exports = db;

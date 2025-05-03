const cheerio = require("cheerio");
const axios = require("axios");
const { findSiteByUrl, createSite } = require("../models/site");
const { findOrCreateVisitor } = require("../models/visiter");
const { createEvent } = require("../models/event");

async function createUserAndEvent(site, req) {
  const { token, type, type_value, timestamp } = req.body;
  const visitor = await findOrCreateVisitor(token, site.id);

  const val = typeof type_value === "object" ? JSON.stringify(type_value) : type_value

  if (visitor) {
    createEvent(visitor.id, type, val, timestamp);
  }
}
// Контроллер для проверки существования сайта с указанным URL и добавления его при необходимости
const AddEvent = async (req, res) => {
  const url = process.env.is_dev
    ? "http://127.0.0.1:5500/test.html"
    : (req.get("Referer") || req.headers.referer);

  const site = await findSiteByUrl(url);

  if (site) {
    await createUserAndEvent(site, req);
  } else {
    const response = await axios.get(url); // Получаем содержимое страницы по URL
    const $ = cheerio.load(response.data); // Используем cheerio для анализа HTML
    const title = $("title").text();
    const description = $("description").text();
    const faviconUrl = $("link[rel='icon']").attr("href");

    if (await createSite(url, title, description, faviconUrl)) {
      await createUserAndEvent(site, req);
      res.status(200).send({ ok: true });
    } else {
      res.status(300).send({ ok: false });
    }
  }
};

module.exports = { AddEvent };

const express = require('express');
const detailsController = require('../controllers/detailsController');
const router = express.Router();
// Обработчик GET запросов для получения записи по id
router.get('/:site_id', detailsController.getSiteById);

module.exports = router;
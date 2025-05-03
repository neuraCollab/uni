const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const db = require('./models/db')
const analyticsRoutes = require('./routes/analytics');
const collectRoutes = require('./routes/collects');
const filtersRoutes = require('./routes/filters');
const mainRouter = require('./routes/main');
const detailRoutes = require('./routes/details');

require('dotenv').config();

const app = express();
app.use(cors());
app.use(bodyParser.json());


// Подключаем маршруты аналитики
app.use('/collect', collectRoutes);
app.use('/detail', detailRoutes);
app.use('/main', mainRouter)
// app.use('/filter', filtersRoutes);
app.post('/print', (req, res) => {
    req.originalUrl
    console.log(req.body);
    res.status(200).send('OK');
})

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

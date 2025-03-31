const {getRefererData} = require('./general/visitsByReference')
const { genderTo3dPiechart } = require('./plots/gender')
const {deviceColumnDiagram} = require('./plots/device')
const {browserPieDiagram} = require('./plots/browser')
const {dailyVisitsAndReferrerStats} = require("./details/dailyVisitsAndReferrerStats") 
const {getAnalyticsData} = require("./details/generalInformation") 
const {getMostUsedMetrics} = require("./details/getMostUsedMetrics") 

exports.getSiteById = async (req, res) => {

    const site_id = req.params.site_id
    // const data = req.body;
    const data = {
        genderTo3dPiechart: await genderTo3dPiechart(site_id),
        deviceColumnDiagram: await deviceColumnDiagram(site_id),
        browserPieDiagram: await browserPieDiagram(site_id),
        refererData: await getRefererData(site_id),
        dailyVisitsAndReferrerStats: await dailyVisitsAndReferrerStats(site_id),
        analyticsData: await getAnalyticsData(site_id),
        getMostUsedMetrics: await getMostUsedMetrics(site_id)
    }
    res.send({data: data})
  
};
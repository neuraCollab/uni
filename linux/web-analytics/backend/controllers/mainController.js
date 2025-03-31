const {getLastThreeSitesInfo} = require('./general/lastThreeSitesInfo')
const {getVisitsByDate} = require('./general/visitsByDate')
const {getRefererData} = require('./general/visitsByReference')
const { genderTo3dPiechart } = require('./plots/gender')
const {geolocationPlot} = require('./plots/geolocation')
const {deviceColumnDiagram} = require('./plots/device')
const {browserPieDiagram} = require('./plots/browser')
const {getTotalTimeOnSite} = require('./plots/TimeByGender')

exports.getAllData = async (req, res) => {
    // const data = req.body;
    const data = {
        lastThreeSitesInfo: await getLastThreeSitesInfo(),
        genderTo3dPiechart: await genderTo3dPiechart(),
        geolocationPlot: await geolocationPlot(),
        deviceColumnDiagram: await deviceColumnDiagram(),
        browserPieDiagram: await browserPieDiagram(),
        visitsByDate: await getVisitsByDate(),
        refererData: await getRefererData(),
        timeByGender: await getTotalTimeOnSite()
    }
    res.send({data: data})
  
};
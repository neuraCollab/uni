import React from "react";
import { Chart } from "react-google-charts";

// export const data = [
//   ["Element", "Density"],
//   ["Copper", 8.94, ], // RGB value
//   ["Silver", 10.49], // English color name
//   ["Gold", 19.3],
//   ["Platinum", 21.45], // CSS-style declaration
// ];

function ColumnsChart({data}) {
  return (
    <Chart chartType="ColumnChart" width="100%" height="400px" data={data} />
  );
}
export default ColumnsChart  
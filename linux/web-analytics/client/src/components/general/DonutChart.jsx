import React from "react";
import { Chart } from "react-google-charts";


export const options = {
  title: "Частота использования браузеров",
  pieHole: 0.4,
  is3D: false,
};

 function DonutPieChart({data}) {
  return (
    <Chart
      chartType="PieChart"
      width="100%"
      height="400px"
      data={data}
      options={options}
    />
  );
}
export default DonutPieChart
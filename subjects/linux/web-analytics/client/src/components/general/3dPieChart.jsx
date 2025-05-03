import React from "react";
import { Chart } from "react-google-charts";


export const options = {
  title: "Частота посещений сайта женщинами/мужчинами",
  is3D: true,
};

export function PieChart({data, option}) {
  return (
    <Chart
      chartType="PieChart"
      data={data}
      options={{...options, ...option}}
      width={"100%"}
      height={"400px"}
    />
  );
}
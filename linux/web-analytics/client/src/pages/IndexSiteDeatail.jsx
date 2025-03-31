import React, { useState, useEffect } from "react";
import { PieChart } from "../components/general/3dPieChart";
import SomeSitesLineChart from "../components/main/someSitesLineChart";
import DonutPieChart from "../components/general/DonutChart";
import { MultipleLineChart } from "../components/general/MultipleLineChart";
import { useParams } from "react-router-dom";
import axios from "axios";
import ColumnsChart from "../components/site_details/columnsChart";

const IndexSiteDetail = (props) => {
  const { site_id } = useParams();
  const [user, setUser] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      // Замените URL на ваш конкретный endpoint
      const response = await axios.get(
        "http://localhost:3001/detail/" + site_id
      );
      setUser(response.data.data);
    };
    fetchData();
  }, []);

  if (!user.dailyVisitsAndReferrerStats) return <>Loading...</>;

  return (
    <div className="">
      <h1 className="text-4xl font-bold mb-8 text-center">
        Вся информация для Сайта с ID: {site_id} хранится на сервере
      </h1>
      <div className="flex flex-col justify-center gap-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">
              Общее количество событий и их название
            </h2>
            <ul>
              {user.dailyVisitsAndReferrerStats.referrerStats.map(
                (el, index) => (
                  <li key={index} className="text-gray-700 mb-2">
                    {el.eventType}: {el.count}
                  </li>
                )
              )}
            </ul>
          </div>
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">
              Общие характеристики устройств пользователей
            </h2>
            <ul>
              {Object.entries(user.getMostUsedMetrics).map(
                ([key, value], index) => (
                  <li key={index} className="text-gray-700 mb-2">
                    {key}: {value}
                  </li>
                )
              )}
            </ul>
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-md p-6">
          <PieChart data={user.genderTo3dPiechart} />
        </div>
        <div className="bg-white rounded-lg shadow-md p-6">
          <SomeSitesLineChart />
        </div>
        <div className="bg-white rounded-lg shadow-md p-6">
          <DonutPieChart data={user.browserPieDiagram} />
        </div>
        <div className="bg-white rounded-lg shadow-md p-6">
          <ColumnsChart data={user.refererData} />
        </div>
        <div className="bg-white rounded-lg shadow-md p-6">
          <MultipleLineChart />
        </div>
        <div className="bg-white rounded-lg shadow-md p-6">
          <PieChart
            option={{
              title: "С каких устройств чаще всего заходят пользователи",
            }}
            data={user.deviceColumnDiagram}
          />
        </div>
      </div>
    </div>
  );
};

export default IndexSiteDetail;

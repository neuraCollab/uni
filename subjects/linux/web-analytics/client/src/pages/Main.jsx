import React, { useState, useEffect } from 'react';
import MapChart from "../components/main/MapChart";
import SomeSitesLineChart from "../components/main/someSitesLineChart";
import { PieChart } from "../components/general/3dPieChart";
import DonutPieChart from "../components/general/DonutChart";
import { MultipleLineChart } from "../components/general/MultipleLineChart";
import AdvansedLayout from "../components/main/AdvansedLayout";
import axios from 'axios';

const Main = () => {

  const [user, setUser] = useState([]);

  console.log(user);
  useEffect(() => {
    const fetchData = async () => {
        // Замените URL на ваш конкретный endpoint
        const response = await axios.get('http://localhost:3001/main');
        setUser(response.data.data);
    }
    fetchData();
  }, []);

  if(!user) return <></>

  return (
    <div className="">
      <div className="flex flex-col lg:flex-row lg:space-x-6">
        <div className="flex-1 mb-6 lg:mb-0">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">
              Расширенный Интерфейс Аналитики
            </h2>
            <AdvansedLayout lastThreeSitesInfo={user.lastThreeSitesInfo} />
            <p className="text-gray-700">
              Этот модуль представляет собой передовую платформу для
              визуализации и анализа данных, позволяющую пользователям в
              реальном времени отслеживать активность и взаимодействие на своих
              веб-сайтах. С помощью данного инструмента вы можете получить
              глубокое понимание поведения пользователей, оптимизировать свои
              ресурсы и улучшить общую эффективность веб-сайта.
            </p>
          </div>
        </div>
        <div className="flex-1">
          <div className="bg-white rounded-lg shadow-md p-6 max-w-max">
            <PieChart data={user.genderTo3dPiechart} />
            <p className="text-gray-700">
              Наша 3D круговая диаграмма предоставляет динамичное и
              интерактивное представление данных, позволяя вам наглядно оценить
              различные метрики, такие как процентное распределение типов
              пользователей или демографические показатели. Используйте эту
              диаграмму для быстрой оценки и сравнения различных сегментов вашей
              аудитории.
            </p>
          </div>
        </div>
      </div>
      <div className="mt-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-6xl font-bold text-center mb-4">
            Интерактивная Карта Взаимодействий
          </h2>
          <MapChart data={user.geolocationPlot} />
          <p className="text-gray-700">
            Используйте нашу интерактивную карту для визуализации
            географического распределения ваших пользователей. Этот инструмент
            позволяет вам видеть, откуда приходят ваши посетители, что помогает
            при планировании маркетинговых кампаний и оптимизации контента для
            различных регионов.
          </p>
        </div>
      </div>
      <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <SomeSitesLineChart data={user.visitsByDate} />
          <p className="text-gray-700">
            Наш линейный график активности сайтов иллюстрирует тенденции и
            изменения во времени, предоставляя важную информацию о пиковых и
            спадающих периодах активности. Используйте этот график для
            мониторинга эффективности проведенных изменений на сайте или для
            анализ
          </p>
        </div>
        <div className="bg-white rounded-lg shadow-md p-6">
          <MultipleLineChart data={user.timeByGender} />
          <p className="text-gray-700">
            Множественный линейный график предоставляет сравнительный анализ
            между различными сайтами или различными параметрами одного сайта. С
            его помощью вы можете одновременно наблюдать за несколькими
            метриками, что позволяет выявить корреляции и зависимости между
            различными аспектами вашего веб-присутствия.
          </p>
        </div>
      </div>
      <div className="mt-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <DonutPieChart data={user.browserPieDiagram} />
          <p className="text-gray-700">
            Круговая диаграмма в форме пончика предлагает ещё один способ
            визуализации данных, делая акцент на пропорциональных отношениях
            внутри выбранной категории. Этот стиль диаграммы особенно полезен
            для демонстрации составных частей целого, например, доли различных
            источников трафика или вклада различных каналов в общий доход.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Main;

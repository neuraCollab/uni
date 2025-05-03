import { List, Avatar } from "flowbite-react";
import { Link } from "react-router-dom";

export default function AdvansedLayout({ lastThreeSitesInfo = [] }) {

  return (
    <List unstyled className=" divide-y divide-gray-200 dark:divide-gray-700">
      {lastThreeSitesInfo.map((el) => {
        return (
          <Link
            to={`/details/${el.id}`}
            className="text-blue-500 hover:underline"
          >
            <List.Item className="pb-3 sm:pb-4">
              <div className="flex items-center space-x-4 rtl:space-x-reverse">
                <Avatar
                  img={el.faviconUrl || "/avatar.jpg"}
                  alt="Neil image"
                  rounded
                  size="sm"
                />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-gray-900 dark:text-white">
                    <a href={el.url || "/"}> {el.url || "Ссылка на сайт"}</a>
                  </p>
                  <p className="truncate text-sm text-gray-500 dark:text-gray-400">
                    {el.description || "Описание сайта"}
                  </p>
                </div>
                <div className="inline-flex items-center text-base font-semibold text-gray-900 dark:text-white">
                  {el.title || "Название сайта"}
                </div>
              </div>
            </List.Item>
            {/* Content of your list item */}
          </Link>
        );
      })}
    </List>
  );
}

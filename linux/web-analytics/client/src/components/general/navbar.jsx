
"use client";

import { Button, Navbar } from "flowbite-react";

export default function SiteNavBar() {
  return (
    <Navbar fluid rounded>
      <Navbar.Brand href="/">
        <img src="/favicon.ico" className="mr-3 h-6 sm:h-9" alt="Flowbite React Logo" />
        <span className="self-center whitespace-nowrap text-xl font-semibold dark:text-white">Аналитика Онлайн</span>
      </Navbar.Brand>
      <div className="flex md:order-2">
        <Button>Просто кнопка</Button>
        <Navbar.Toggle />
      </div>
      <Navbar.Collapse>
        <Navbar.Link href="/" active>
        Главная
        </Navbar.Link>
        <Navbar.Link href="/details/1">Пример страницы деталей</Navbar.Link>
      </Navbar.Collapse>
    </Navbar>
  );
}

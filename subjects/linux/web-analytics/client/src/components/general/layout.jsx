import React from "react";
import { Outlet } from "react-router-dom";
import SiteFooter from "./Footer";
import SiteNavBar from "./navbar";

const Layout = () => {
  return (
    <>
      <main className="max-w-screen-2xl mx-auto p-6">
        <SiteNavBar />
        <Outlet />
        <SiteFooter />
      </main>
    </>
  );
};

export default Layout;

import {
  createBrowserRouter,

} from "react-router-dom";
import App from "../App";
import IndexSiteDetail from "../pages/IndexSiteDeatail";
import Main from "../pages/Main";


export const router = createBrowserRouter([
    {
      path: "/",
      element: (
        <App /> 
      ),
      children: [
        {
          path: "",
          element: (
            <Main />
          )
        },
        {
          path: "details/:site_id",
          element: (
            <IndexSiteDetail />
          )
        }
      ]
    },
   
  ]);
use yew::prelude::*;
use yew_router::prelude::*;

use crate::router::{switch, Route};

#[function_component(App)]
pub fn app() -> Html {
    html! {
        <main>
            <div id="header">
                <h1 id="header-title">{"Learning machine learning"}</h1>
            </div>
            <BrowserRouter>
                <Switch<Route> render={switch}/>
            </BrowserRouter>
        </main>
    }
}

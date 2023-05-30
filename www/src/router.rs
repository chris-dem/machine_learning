use yew::prelude::*;
use yew_router::prelude::*;

use crate::home::Home;

#[derive(Clone, Routable, PartialEq)]
pub enum Route {
    #[at("/")]
    Home,
    #[at("/contents/*")]
    Contents,
}

pub fn switch(routes: Route) -> Html {
    match routes {
        Route::Home => html! {
            <Home />
        },
        Route::Contents => html! {
            <>
            </>
        },
    }
}

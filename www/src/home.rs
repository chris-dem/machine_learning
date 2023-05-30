use crate::router::Route;
use yew::prelude::*;
use yew_router::prelude::*;

#[function_component(Home)]
pub fn home() -> Html {
    html! {
        <div>
            <h2>{"Motivation"}</h2>
            <p>
                {"This site is a machine learning archive site. It will work as a blog site for all of my findings in the machine learning world and how they work"}
            </p>

            <Link<Route> to={Route::Contents}> {"Click here to go to the list of contents"}</Link<Route>>
        </div>
    }
}

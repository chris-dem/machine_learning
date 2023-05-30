use yew::prelude::*;
use yew_router::prelude::*;

#[derive(Debug, Clone, PartialEq, Routable)]
enum RouterConcepts {
    #[at("/knn")]
    KNN,
}
fn content_switch(route: RouterConcepts) -> Html {
    match route {
        RouterConcepts::KNN => todo!(),
    }
}

#[function_component(MainContent)]
pub fn main_content() -> Html {
    todo!()
}

use std::fmt::Display;

use crate::main_site::contents::main_content::MainContent;
use yew::prelude::*;
use yew_router::prelude::*;

#[derive(Clone, PartialEq, Routable)]
enum ContentRouter {
    #[at("/main_table")]
    Table,
    #[at("/content/*")]
    Content,
}

#[function_component(TableContents)]
fn table_contents() -> Html {
    html! {
        <>
            <h2>{"Table of contents"}</h2>
            <p>
                {"I would suggest going through this conent from top to bottom"}
            </p>
        </>
    }
}

fn table_switch(route: ContentRouter) -> Html {
    match route {
        ContentRouter::Table => html! { <TableContents /> },
        ContentRouter::Content => html! { <MainContent/> },
    }
}

#[function_component(Content)]
pub fn content() -> Html {
    html! {
        <>
            <BrowserRouter>
                <Switch<ContentRouter> render={table_switch}/>
            </BrowserRouter>
        </>
    }
}

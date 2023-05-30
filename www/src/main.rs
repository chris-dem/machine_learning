mod app;
mod main_site;
mod home;
mod router;

fn main() {
    yew::Renderer::<app::App>::new().render();
}

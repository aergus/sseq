use ext::utils::construct_from_json;
use serde_json::json;

#[test]
fn negative_min_degree() {
    let mut json = json!({
        "type": "finite dimensional module",
        "p": 2,
        "generic": false,
        "gens": {"x0": -2},
        "actions": [],
        "products": [{"hom_deg":3,"int_deg":11,"class":[1],"name":"c_0"}],
    });
    let resolution = construct_from_json(&mut json, "adem").unwrap();

    resolution.resolve_through_bidegree(20, 20);
}

#[test]
fn positive_min_degree() {
    let mut json = json!({
        "type": "finite dimensional module",
        "p": 2,
        "generic": false,
        "gens": {"x0": 2},
        "actions": [],
        "products":[{"hom_deg":3,"int_deg":11,"class":[1],"name":"c_0"}],
    });
    let resolution = construct_from_json(&mut json, "adem").unwrap();

    resolution.resolve_through_bidegree(20, 20);
}

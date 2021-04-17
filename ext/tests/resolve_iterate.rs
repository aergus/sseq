use ext::chain_complex::{ChainComplex, FreeChainComplex};
use ext::utils::construct_from_json;
use ext::utils::load_module_json;
use ext::utils::Config;

#[test]
fn resolve_iterate() {
    for name in &["S_2", "S_3", "Ceta", "Calpha", "C3", "Joker"] {
        let config = Config {
            module_file_name: (*name).to_string(),
            algebra_name: String::from("milnor"),
        };
        test_iterate(&config);

        let config = Config {
            module_file_name: (*name).to_string(),
            algebra_name: String::from("adem"),
        };
        test_iterate(&config);
    }
}

#[allow(clippy::redundant_clone)]
fn test_iterate(config: &Config) {
    println!(
        "Resolving {} with {} basis",
        &config.module_file_name, &config.algebra_name
    );

    let json = load_module_json(&config.module_file_name).unwrap();

    let first = construct_from_json(&mut json.clone(), &config.algebra_name).unwrap();
    let second = construct_from_json(&mut json.clone(), &config.algebra_name).unwrap();

    first.compute_through_bidegree(20, 20);

    second.compute_through_bidegree(0, 0);
    second.compute_through_bidegree(5, 5);
    second.compute_through_bidegree(10, 7);
    second.compute_through_bidegree(7, 10);
    second.compute_through_bidegree(18, 18);
    second.compute_through_bidegree(14, 14);
    second.compute_through_bidegree(15, 15);
    second.compute_through_bidegree(20, 20);

    assert_eq!(
        first.graded_dimension_string(),
        second.graded_dimension_string()
    );

    #[cfg(feature = "concurrent")]
    {
        let bucket = thread_token::TokenBucket::new(2);
        let third = construct_from_json(&mut json.clone(), &config.algebra_name).unwrap();

        third.compute_through_bidegree_concurrent(0, 0, &bucket);
        third.compute_through_bidegree_concurrent(5, 5, &bucket);
        third.compute_through_bidegree_concurrent(10, 7, &bucket);
        third.compute_through_bidegree_concurrent(7, 10, &bucket);
        third.compute_through_bidegree_concurrent(18, 18, &bucket);
        third.compute_through_bidegree_concurrent(14, 14, &bucket);
        third.compute_through_bidegree_concurrent(15, 15, &bucket);
        third.compute_through_bidegree_concurrent(20, 20, &bucket);

        assert_eq!(
            first.graded_dimension_string(),
            third.graded_dimension_string()
        );
    }
}

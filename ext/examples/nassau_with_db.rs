/// This is essentially a copy of `resolve_through_stem` using a database instead of files.
///
/// This is for testing purposes only, and should be removed before merging the database feature
/// branch into `master`.
use ext::{chain_complex::FreeChainComplex, save::SaveBackend, utils, utils::construct_nassau};

use r2d2_postgres::{postgres, r2d2, PostgresConnectionManager};

fn main() -> anyhow::Result<()> {
    let (name, module): (String, utils::Config) = query::with_default("Module", "S_2", |s| {
        Result::<_, anyhow::Error>::Ok((s.to_owned(), s.try_into()?))
    });

    let save_target = query::optional(
        "Connection string in the sense of https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING",
        |s|  {
            let manager = PostgresConnectionManager::new(
                s.parse().unwrap(),
                postgres::NoTls,
            );
            r2d2::Pool::new(manager).map(SaveBackend::Database)
        }
    );

    let mut res = construct_nassau(module, save_target)?;
    res.set_name(name);

    let max_n = query::with_default("Max n", "30", str::parse);
    let max_s = query::with_default("Max s", "15", str::parse);

    res.compute_through_stem(max_s, max_n);

    println!("{}", res.graded_dimension_string());

    Ok(())
}

use arrow::array::Int64Array;
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::error::Result;
use datafusion::physical_plan::expressions::Column;
use datafusion::physical_plan::joins::{HashJoinExec, PartitionMode, SortMergeJoinExec};
use datafusion::physical_plan::memory::{LazyBatchGenerator, LazyMemoryExec};
use datafusion::physical_plan::{ExecutionPlan, PhysicalExpr, collect};
use datafusion::prelude::SessionContext;
use parking_lot::RwLock;
use rand::Rng;
use rand::thread_rng;
use std::sync::Arc;
use tokio::runtime::Runtime;

#[derive(Debug)]
struct BatchGenerator {
    batches: Vec<RecordBatch>,
    next_batch_idx: usize,
}

impl std::fmt::Display for BatchGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BatchGenerator(batches: {}, next: {})",
            self.batches.len(),
            self.next_batch_idx
        )
    }
}

impl LazyBatchGenerator for BatchGenerator {
    fn generate_next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.next_batch_idx < self.batches.len() {
            let batch = &self.batches[self.next_batch_idx];
            self.next_batch_idx += 1;
            Ok(Some(batch.clone()))
        } else {
            Ok(None)
        }
    }
}

fn create_lazy_memory_exec(
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let generator = BatchGenerator {
        batches,
        next_batch_idx: 0,
    };
    let wrapped_generator: Arc<RwLock<dyn LazyBatchGenerator>> = Arc::new(RwLock::new(generator));
    let partitions = vec![wrapped_generator];
    Ok(Arc::new(LazyMemoryExec::try_new(schema, partitions)?))
}

fn join_benchmark(c: &mut Criterion) -> Result<()> {
    let rt = Runtime::new().unwrap();

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

    let batch_sizes = [1024, 1024 * 4, 1024 * 10];
    let num_batches = 5;

    for batch_size in batch_sizes.iter() {
        let batch_size = *batch_size;
        let total_rows = batch_size * num_batches;

        let ctx = SessionContext::new();
        let task_ctx = ctx.task_ctx();

        let mut unsorted_batches_l = Vec::with_capacity(num_batches);
        let mut unsorted_batches_r = Vec::with_capacity(num_batches);
        let mut sorted_batches_l = Vec::with_capacity(num_batches);
        let mut sorted_batches_r = Vec::with_capacity(num_batches);

        let mut rng = thread_rng();

        for i in 0..num_batches {
            let start_key_unsorted = (i * batch_size / 2) as i64;

            let unsorted_id_l = Arc::new(Int64Array::from_iter_values((0..batch_size).map(|_| {
                rng.gen_range(start_key_unsorted..start_key_unsorted + batch_size as i64)
            })));
            let unsorted_id_r = Arc::new(Int64Array::from_iter_values((0..batch_size).map(|_| {
                rng.gen_range(start_key_unsorted..start_key_unsorted + batch_size as i64)
            })));

            unsorted_batches_l
                .push(RecordBatch::try_new(schema.clone(), vec![unsorted_id_l]).unwrap());
            unsorted_batches_r
                .push(RecordBatch::try_new(schema.clone(), vec![unsorted_id_r]).unwrap());

            let start_key_sorted = (i * batch_size) as i64;

            let sorted_id_l = Arc::new(Int64Array::from_iter_values(
                start_key_sorted..start_key_sorted + batch_size as i64,
            ));
            let sorted_id_r = Arc::new(Int64Array::from_iter_values(
                start_key_sorted..start_key_sorted + batch_size as i64,
            ));

            sorted_batches_l.push(RecordBatch::try_new(schema.clone(), vec![sorted_id_l]).unwrap());
            sorted_batches_r.push(RecordBatch::try_new(schema.clone(), vec![sorted_id_r]).unwrap());
        }

        let unsorted_left_exec = create_lazy_memory_exec(schema.clone(), unsorted_batches_l)?;
        let unsorted_right_exec = create_lazy_memory_exec(schema.clone(), unsorted_batches_r)?;
        let sorted_left_exec = create_lazy_memory_exec(schema.clone(), sorted_batches_l)?;
        let sorted_right_exec = create_lazy_memory_exec(schema.clone(), sorted_batches_r)?;

        let join_on: Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)> = vec![(
            Arc::new(Column::new("id", 0)),
            Arc::new(Column::new("id", 0)),
        )];
        let join_type = datafusion::logical_expr::JoinType::Inner;

        let hash_join_unsorted = Arc::new(
            HashJoinExec::try_new(
                unsorted_left_exec.clone(),
                unsorted_right_exec.clone(),
                join_on.clone(),
                None,
                &join_type,
                None,
                PartitionMode::CollectLeft,
                false,
            )
            .unwrap(),
        );

        let smj_join = Arc::new(
            SortMergeJoinExec::try_new(
                sorted_left_exec.clone(),
                sorted_right_exec.clone(),
                join_on.clone(),
                None, // Filter
                join_type,
                vec![SortOptions::default()], // Use imported SortOptions
                false,                        // null_equals_null
            )
            .unwrap(),
        );

        let mut group = c.benchmark_group(format!(
            "Join Benchmark ({} batches x {} rows = {} total rows)",
            num_batches, batch_size, total_rows
        ));
        group.throughput(criterion::Throughput::Elements(total_rows as u64));

        group.bench_function("HashJoin (Unsorted Input)", |b| {
            b.iter(|| {
                rt.block_on(async {
                    let plan = hash_join_unsorted.clone();
                    collect(plan, task_ctx.clone()).await.unwrap()
                })
            })
        });

        group.bench_function("SortMergeJoin (Sorted Input)", |b| {
            b.iter(|| {
                rt.block_on(async {
                    let plan = smj_join.clone();
                    collect(plan, task_ctx.clone()).await.unwrap()
                })
            });
        });

        group.finish();
    }

    Ok(())
}

fn run_benchmarks(c: &mut Criterion) {
    if let Err(e) = join_benchmark(c) {
        eprintln!("Error setting up benchmark: {}", e);
    }
}

criterion_group!(benches, run_benchmarks);
criterion_main!(benches);

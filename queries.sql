-- Q1: WHERE then HAVING
SELECT customer_id 
FROM orders 
WHERE status = 'completed' AND order_amount > 500 
GROUP BY customer_id 
HAVING SUM(order_amount) > 8000;

-- Q2: Three-table join revenue by city and category
SELECT c.city, p.category, SUM(o.order_amount) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category;

-- Q3: Scalar subquery for products above average catalogue price
SELECT product_name 
FROM products 
WHERE price_inr > (SELECT AVG(price_inr) FROM products);

-- Q4: LEFT JOIN for customers with no completed orders
SELECT c.customer_id 
FROM customers c
LEFT JOIN (SELECT DISTINCT customer_id FROM orders WHERE status = 'completed') o ON c.customer_id = o.customer_id
WHERE o.customer_id IS NULL;

-- Q5: CTEs for customers spending above the average spend
WITH cust_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders WHERE status = 'completed' GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) AS mean_spend FROM cust_spend
)
SELECT cs.customer_id 
FROM cust_spend cs, avg_spend av 
WHERE cs.total_spend > av.mean_spend;

-- Q6: Ranking window functions partitioned by city
WITH cust_city_spend AS (
    SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM orders o JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed' GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, total_spend,
       RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) as rnk,
       DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) as dense_rnk,
       ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC) as row_num
FROM cust_city_spend;

-- Q7: Running total of daily completed revenue
SELECT SUBSTR(order_ts, 1, 10) as day,
       SUM(order_amount) AS daily_revenue,
       SUM(SUM(order_amount)) OVER (ORDER BY SUBSTR(order_ts, 1, 10)) as cumulative_revenue
FROM orders
WHERE status = 'completed'
GROUP BY day;
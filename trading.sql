--
-- PostgreSQL database dump
--

-- Dumped from database version 12.2 (Ubuntu 12.2-4)
-- Dumped by pg_dump version 12.2 (Ubuntu 12.2-4)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: historical_usdjpy; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.historical_usdjpy (
    index bigint,
    "time" timestamp with time zone,
    "openBid" double precision,
    "openAsk" double precision,
    "highBid" double precision,
    "highAsk" double precision,
    "lowBid" double precision,
    "lowAsk" double precision,
    "closeBid" double precision,
    "closeAsk" double precision,
    volume bigint,
    complete boolean
);


--
-- Name: trades; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.trades (
    index bigint,
    account text,
    account_type text,
    ask double precision,
    bid double precision,
    instrument text,
    model text,
    prediction_used text,
    probability text,
    stop_loss text,
    take_profit bigint,
    "time" timestamp without time zone,
    trade bigint
);

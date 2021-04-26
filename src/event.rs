use std::collections::VecDeque;
use std::thread;
use std::thread::JoinHandle;
use std::sync::{mpsc, Mutex, Arc};
use std::sync::mpsc::{Sender, Receiver};


pub trait Subscriber<E>: Send {
    fn notify(&self, e: E);
}

pub struct FnSubscriber<E> {
    f: Box<dyn Fn(E) -> ()>
}

impl<F: 'static, E> From<F> for FnSubscriber<E> where F: Fn(E) -> () {
    fn from(f: F) -> Self {
        Self {
            f: Box::new(f)
        }
    }
}

// impl<E> Subscriber<E> for FnSubscriber<E> {
//     fn notify(&self, e: E) {
//         self.f(e);
//     }
// }

pub struct EventManager<E> {
    manager_handle: Option<JoinHandle<()>>,
    subscribers: Arc<Mutex<Vec<Box<dyn Subscriber<E>>>>>,
    event_queue: Sender<Option<E>>
}

// Ideally the EventManager would first set up subscribers then say post an event and when the event is called all subscribers get notified

// mpsc means multiple producer single consumer
// Clone or Copy?
impl<E: 'static + Send + Clone> EventManager<E> {
    pub fn new() -> Self {
        let subscribers: Arc<Mutex<Vec<Box<dyn Subscriber<E>>>>> = Arc::new(Mutex::new(Vec::new()));
        let sub_clone = subscribers.clone();
        let (tx, rx): (Sender<Option<E>>, Receiver<Option<E>>) = mpsc::channel();
        let manager_handle = thread::spawn(move || {
            loop {
                let event = rx.recv();
                if event.is_err() {
                    return;
                }
                let event = event.unwrap();
                match event {
                    None => {
                        return;
                    },
                    Some(event) => {
                        let subs = sub_clone.lock().unwrap();
                        if !subs.is_empty() {
                            for i in subs.iter() {
                                i.notify(event.clone());
                            }
                        }
                    }
                }
            }
        });
        EventManager {
            subscribers,
            manager_handle: Some(manager_handle),
            event_queue: tx
        }
    }
    pub fn subscribe<S: 'static>(&mut self, f: S) where S: Subscriber<E> {
        self.subscribers.lock().unwrap().push(Box::new(f));
    }

    pub fn throw_event(&self, e: E) {
        self.event_queue.send(Some(e));
    }
}

impl<E> Drop for EventManager<E> {
    fn drop(&mut self) {
        self.event_queue.send(None);
        self.manager_handle.take().unwrap().join().expect("Error with Joining");
    }
}



#[cfg(test)]
mod tests {
    use crate::event::{EventManager, FnSubscriber};

    #[test]
    fn event_test() {
        // let mut manager = EventManager::new();
        // let subscriber = FnSubscriber::from(|e: i32| {
        //     println!("Inside Subscriber: {}", e);
        // });
        // manager.subscribe(subscriber);
        //
        // manager.throw_event(23);
    }
}
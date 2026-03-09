
-- Profiles table for user info
CREATE TABLE public.profiles (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
  display_name TEXT,
  email TEXT,
  role TEXT NOT NULL DEFAULT 'user',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile" ON public.profiles FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own profile" ON public.profiles FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own profile" ON public.profiles FOR UPDATE USING (auth.uid() = user_id);

-- Auto-create profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (user_id, email, display_name)
  VALUES (NEW.id, NEW.email, COALESCE(NEW.raw_user_meta_data->>'display_name', split_part(NEW.email, '@', 1)));
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = public;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- FAQ table (publicly readable)
CREATE TABLE public.faq (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  category TEXT DEFAULT 'General',
  sort_order INT DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

ALTER TABLE public.faq ENABLE ROW LEVEL SECURITY;
CREATE POLICY "FAQ is publicly readable" ON public.faq FOR SELECT USING (true);

-- Support tickets
CREATE TABLE public.support_tickets (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  subject TEXT NOT NULL,
  description TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'open',
  priority TEXT NOT NULL DEFAULT 'medium',
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

ALTER TABLE public.support_tickets ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own tickets" ON public.support_tickets FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can create own tickets" ON public.support_tickets FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own tickets" ON public.support_tickets FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete own tickets" ON public.support_tickets FOR DELETE USING (auth.uid() = user_id);

-- Ticket replies
CREATE TABLE public.ticket_replies (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  ticket_id UUID NOT NULL REFERENCES public.support_tickets(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  is_staff BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

ALTER TABLE public.ticket_replies ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view replies on own tickets" ON public.ticket_replies FOR SELECT
  USING (EXISTS (SELECT 1 FROM public.support_tickets WHERE id = ticket_id AND user_id = auth.uid()));
CREATE POLICY "Users can reply to own tickets" ON public.ticket_replies FOR INSERT
  WITH CHECK (auth.uid() = user_id AND EXISTS (SELECT 1 FROM public.support_tickets WHERE id = ticket_id AND user_id = auth.uid()));

-- Timestamp trigger for tickets
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SET search_path = public;

CREATE TRIGGER update_tickets_updated_at
  BEFORE UPDATE ON public.support_tickets
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- Seed FAQ data
INSERT INTO public.faq (question, answer, category, sort_order) VALUES
('How does CropVision predict crop yield?', 'CropVision uses satellite imagery (Landsat & Sentinel), NDVI analysis, weather data, and machine learning models (Random Forest & CNN) to predict crop yield in tons per hectare.', 'General', 1),
('What satellite data sources are used?', 'We use Landsat 8/9 and Sentinel-2 satellite imagery, extracting Red and Near-Infrared (NIR) bands for vegetation monitoring and NDVI calculation.', 'Technology', 2),
('What is NDVI and how is it used?', 'NDVI (Normalized Difference Vegetation Index) measures vegetation health using the formula (NIR - Red) / (NIR + Red). Values closer to 1 indicate healthy vegetation.', 'Technology', 3),
('How accurate are the predictions?', 'Our ML models achieve R² scores of 0.89-0.94 depending on the crop type and region, with continuous improvement as more data becomes available.', 'Accuracy', 4),
('Which crops are supported?', 'CropVision supports 40+ crops including Rice, Wheat, Maize, Soybean, Cotton, Sugarcane, and many more across all Indian states.', 'Coverage', 5),
('How often is satellite data updated?', 'Sentinel-2 data is updated every 5 days and Landsat every 16 days. Weather data is updated in real-time from IMD and OpenWeather.', 'Data', 6),
('Can I export my prediction reports?', 'Yes, all prediction data, charts, and reports can be exported as PDF or CSV from the Dashboard.', 'Features', 7),
('How do I contact support?', 'You can submit a support ticket through this Help Desk, use the AI assistant for instant answers, or email us at support@cropvision.ai.', 'Support', 8);
